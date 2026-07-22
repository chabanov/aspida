with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Environment_Variables;
with Ada.Directories;
with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;
with Ada.Streams; use Ada.Streams;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;

package body LLM_ImgGen is

   --  ---- C API of libaspida_imggen.so (linked at build time) --------------
   function C_Init (Dit, Vae, Llm, Mmproj : chars_ptr) return int
     with Import, Convention => C, External_Name => "aspida_img_init";
   function C_Generate
     (Prompt, Ref_Path : chars_ptr;
      W, H, Steps : int; Cfg : C_float; Seed : Long_Long_Integer;
      Out_Path : chars_ptr) return int
     with Import, Convention => C, External_Name => "aspida_img_generate";

   --  ---- model paths (env overrides, /opt/sdmodels defaults) --------------
   function Env (Name, Default : String) return String is
     (if Ada.Environment_Variables.Exists (Name)
      then Ada.Environment_Variables.Value (Name) else Default);

   Dit_Path : constant String :=
     Env ("ASPIDA_IMG_DIT", "/opt/sdmodels/dit/qwen-image-edit-2511-Q8_0.gguf");
   Vae_Path : constant String :=
     Env ("ASPIDA_IMG_VAE",
          "/opt/sdmodels/vae/split_files/vae/qwen_image_vae.safetensors");
   Llm_Path : constant String :=
     Env ("ASPIDA_IMG_LLM", "/opt/sdmodels/llm/Qwen2.5-VL-7B-Instruct.Q8_0.gguf");
   Mmproj_Path : constant String :=
     Env ("ASPIDA_IMG_MMPROJ",
          "/opt/sdmodels/llm/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf");

   --  ---- lazy, serialised init -------------------------------------------
   protected Init_Lock is
      entry Acquire;
      procedure Release;
   private
      Busy : Boolean := False;
   end Init_Lock;
   protected body Init_Lock is
      entry Acquire when not Busy is begin Busy := True; end Acquire;
      procedure Release is begin Busy := False; end Release;
   end Init_Lock;

   Loaded : Boolean := False with Volatile;

   function Available return Boolean is
   begin
      return Ada.Directories.Exists (Dit_Path)
        and then Ada.Directories.Exists (Vae_Path)
        and then Ada.Directories.Exists (Llm_Path);
   end Available;

   --  Must be called with Init_Lock held.
   function Ensure_Loaded_Locked return Boolean is
   begin
      if Loaded then return True; end if;
      if not Available then return False; end if;
      declare
         D : chars_ptr := New_String (Dit_Path);
         V : chars_ptr := New_String (Vae_Path);
         L : chars_ptr := New_String (Llm_Path);
         M : chars_ptr := New_String (Mmproj_Path);
         Rc : constant int := C_Init (D, V, L, M);
      begin
         Free (D); Free (V); Free (L); Free (M);
         Loaded := (Rc = 0);
      end;
      return Loaded;
   end Ensure_Loaded_Locked;

   function Generate
     (Prompt   : String;
      Ref_Path : String := "";
      Width    : Integer := 1024;
      Height   : Integer := 1024;
      Steps    : Integer := 20;
      Cfg      : Float   := 2.5;
      Seed     : Long_Long_Integer := -1;
      Out_Path : String)
      return Boolean
   is
      Result : Boolean := False;
   begin
      Init_Lock.Acquire;
      begin
         if Ensure_Loaded_Locked then
            declare
               P  : chars_ptr := New_String (Prompt);
               R  : chars_ptr := New_String (Ref_Path);   --  "" => t2i
               O  : chars_ptr := New_String (Out_Path);
               Rc : int;
            begin
               Rc := C_Generate
                 (P, R, int (Width), int (Height), int (Steps),
                  C_float (Cfg), Seed, O);
               Free (P); Free (R); Free (O);
               Result := (Rc = 0);
            end;
         end if;
      exception
         when others => Init_Lock.Release; raise;
      end;
      Init_Lock.Release;
      return Result;
   end Generate;

   --  ---- base64 ----------------------------------------------------------
   Alpha : constant String :=
     "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

   function Encode_File_B64 (Path : String) return String is
      F : File_Type;
      Buf : Stream_Element_Array (1 .. 3);
      Last : Stream_Element_Offset;
      Result : String (1 .. 4 * ((Integer (Ada.Directories.Size (Path)) + 2) / 3));
      RI : Natural := 0;
   begin
      Open (F, In_File, Path);
      while not End_Of_File (F) loop
         Read (Stream (F).all, Buf, Last);
         declare
            N : constant Stream_Element_Offset := Last;   --  1..3 bytes read
            B0 : constant Natural := Natural (Buf (1));
            B1 : constant Natural := (if N >= 2 then Natural (Buf (2)) else 0);
            B2 : constant Natural := (if N >= 3 then Natural (Buf (3)) else 0);
         begin
            RI := RI + 1; Result (RI) := Alpha (B0 / 4 + 1);
            RI := RI + 1; Result (RI) := Alpha ((B0 mod 4) * 16 + B1 / 16 + 1);
            RI := RI + 1;
            Result (RI) := (if N >= 2 then Alpha ((B1 mod 16) * 4 + B2 / 64 + 1) else '=');
            RI := RI + 1;
            Result (RI) := (if N >= 3 then Alpha (B2 mod 64 + 1) else '=');
         end;
      end loop;
      Close (F);
      return Result (1 .. RI);
   end Encode_File_B64;

   function B64_Val (C : Character) return Integer is
   begin
      for I in Alpha'Range loop
         if Alpha (I) = C then return I - 1; end if;
      end loop;
      return -1;   --  padding / whitespace / invalid
   end B64_Val;

   function Decode_B64_To_Temp (B64 : String) return String is
      --  strip an optional "data:...;base64," prefix
      Comma : constant Natural := Index (B64, ",");
      Start : constant Natural :=
        (if Index (B64 (B64'First .. Natural'Min (B64'Last, B64'First + 60)),
                   "base64") > 0 and then Comma > 0
         then Comma + 1 else B64'First);
      Path : constant String :=
        "/tmp/aspida_ref_" & Trim (Integer'Image (B64'Length), Both) & ".png";
      F : File_Type;
      Acc : array (0 .. 3) of Integer := [others => 0];
      NAcc : Natural := 0;
   begin
      Create (F, Out_File, Path);
      for I in Start .. B64'Last loop
         declare
            V : constant Integer := B64_Val (B64 (I));
         begin
            if V >= 0 then
               Acc (NAcc) := V; NAcc := NAcc + 1;
               if NAcc = 4 then
                  declare
                     O0 : constant Stream_Element := Stream_Element (Acc (0) * 4 + Acc (1) / 16);
                     O1 : constant Stream_Element := Stream_Element ((Acc (1) mod 16) * 16 + Acc (2) / 4);
                     O2 : constant Stream_Element := Stream_Element ((Acc (2) mod 4) * 64 + Acc (3));
                  begin
                     Stream_Element'Write (Stream (F), O0);
                     Stream_Element'Write (Stream (F), O1);
                     Stream_Element'Write (Stream (F), O2);
                  end;
                  NAcc := 0;
               end if;
            end if;
         end;
      end loop;
      --  tail (2 or 3 accumulated = 1 or 2 output bytes)
      if NAcc >= 2 then
         Stream_Element'Write (Stream (F), Stream_Element (Acc (0) * 4 + Acc (1) / 16));
         if NAcc = 3 then
            Stream_Element'Write (Stream (F), Stream_Element ((Acc (1) mod 16) * 16 + Acc (2) / 4));
         end if;
      end if;
      Close (F);
      return Path;
   exception
      when others =>
         if Is_Open (F) then Close (F); end if;
         return "";
   end Decode_B64_To_Temp;

end LLM_ImgGen;
