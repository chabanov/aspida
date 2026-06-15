---------------------------------------------------------------------
-- LLM_GGUF body — GGUF v3 parser with POSIX I/O
---------------------------------------------------------------------

with Ada.Text_IO;
with System.Storage_Elements;

package body LLM_GGUF is

   use Ada.Strings.Unbounded;

   --------------------------------------------------------------------
   -- POSIX I/O imports (thin bindings to libSystem)
   --------------------------------------------------------------------
   type C_Int is new Integer;
   type C_Size is mod 2**64;
   type C_Off is new Long_Long_Integer;

   function C_Open  (Path : String; Flags : C_Int; Mode : C_Int) return C_Int
     with Import, Convention => C, External_Name => "open";
   function C_Read  (FD : C_Int; Buf : System.Address; Count : C_Size) return C_Int
     with Import, Convention => C, External_Name => "read";
   function C_LSeek (FD : C_Int; Offset : C_Off; Whence : C_Int) return C_Off
     with Import, Convention => C, External_Name => "lseek";
   function C_Close (FD : C_Int) return C_Int
     with Import, Convention => C, External_Name => "close";

   O_RDONLY : constant C_Int := 0;
   SEEK_SET : constant C_Int := 0;
   SEEK_CUR : constant C_Int := 1;

   --  Explicitly discard the return code of a side-effecting POSIX call
   --  (close/lseek) whose result we intentionally ignore.
   procedure Ignore (Unused : C_Int)  is null;
   procedure Ignore (Unused : C_Off)  is null;

   --------------------------------------------------------------------
   -- Binary reader helpers
   --------------------------------------------------------------------

   --  Read exactly Count bytes, looping over short reads (read() may legally
   --  return fewer bytes than requested — at EOF, on a signal, or on a pipe).
   --  Advances the destination address by the bytes already read.
   procedure Read_Exact (FD : C_Int; Addr : System.Address; Count : Natural) is
      use System.Storage_Elements;
      Remaining : Natural := Count;
      Cur       : System.Address := Addr;
      N         : C_Int;
   begin
      while Remaining > 0 loop
         N := C_Read (FD, Cur, C_Size (Remaining));
         if N <= 0 then
            raise Constraint_Error with "Short read from GGUF file";
         end if;
         Remaining := Remaining - Natural (N);
         Cur := Cur + Storage_Offset (N);
      end loop;
   end Read_Exact;

   type U8_T  is mod 2 ** 8;
   function Read_U8 (FD : C_Int) return U8_T is
      V : U8_T := 0;
   begin
      Read_Exact (FD, V'Address, 1);
      return V;
   end Read_U8;

   type U16_T is mod 2 ** 16;
   function Read_U16 (FD : C_Int) return U16_T is
      V : U16_T := 0;
   begin
      Read_Exact (FD, V'Address, 2);
      return V;
   end Read_U16;

   function Read_U32 (FD : C_Int) return U32 is
      V : U32 := 0;
   begin
      Read_Exact (FD, V'Address, 4);
      return V;
   end Read_U32;

   function Read_U64 (FD : C_Int) return U64 is
      V : U64 := 0;
   begin
      Read_Exact (FD, V'Address, 8);
      return V;
   end Read_U64;

   function Read_F32 (FD : C_Int) return Float is
      V : Float := 0.0;
   begin
      Read_Exact (FD, V'Address, 4);
      return V;
   end Read_F32;

   function Read_Bool (FD : C_Int) return Boolean is
      V : U32;
   begin
      V := Read_U32 (FD);
      return V /= 0;
   end Read_Bool;

   function Read_String (FD : C_Int) return String is
      Len : constant U64 := Read_U64 (FD);
      S   : String (1 .. Natural (Len));
   begin
      if Len > 0 then
         Read_Exact (FD, S'Address, Natural (Len));
      end if;
      return S;
   end Read_String;

   --------------------------------------------------------------------
   -- Metadata value types
   --------------------------------------------------------------------

   type GGUF_Value_Type is
     (GGUF_TYPE_U8,    -- 0
      GGUF_TYPE_I8,    -- 1
      GGUF_TYPE_U16,   -- 2
      GGUF_TYPE_I16,   -- 3
      GGUF_TYPE_U32,   -- 4
      GGUF_TYPE_I32,   -- 5
      GGUF_TYPE_F32,   -- 6
      GGUF_TYPE_BOOL,  -- 7
      GGUF_TYPE_STR,   -- 8
      GGUF_TYPE_ARR,   -- 9
      GGUF_TYPE_U64,   -- 10
      GGUF_TYPE_I64);  -- 11

   function Read_Metadata_Value (FD : C_Int; VT : GGUF_Value_Type) return String is
      V_Str : Unbounded_String;
   begin
      case VT is
         --  Small scalars are 1 or 2 bytes on disk; reading 4 would desync the
         --  file cursor and corrupt every subsequent metadata entry.
         when GGUF_TYPE_U8   => V_Str := To_Unbounded_String (U64'Image (U64 (Read_U8 (FD))));
         when GGUF_TYPE_I8   =>
            declare
               B : constant U8_T := Read_U8 (FD);
            begin
               V_Str := To_Unbounded_String (Integer'Image
                 (if B >= 128 then Integer (B) - 256 else Integer (B)));
            end;
         when GGUF_TYPE_U16  => V_Str := To_Unbounded_String (U64'Image (U64 (Read_U16 (FD))));
         when GGUF_TYPE_I16  =>
            declare
               H : constant U16_T := Read_U16 (FD);
            begin
               V_Str := To_Unbounded_String (Integer'Image
                 (if H >= 32768 then Integer (H) - 65536 else Integer (H)));
            end;
         when GGUF_TYPE_U32  => V_Str := To_Unbounded_String (U64'Image (U64 (Read_U32 (FD))));
         when GGUF_TYPE_I32  => V_Str := To_Unbounded_String (Integer'Image (Integer (Read_U32 (FD))));
         when GGUF_TYPE_F32  => V_Str := To_Unbounded_String (Float'Image (Read_F32 (FD)));
         when GGUF_TYPE_BOOL => V_Str := To_Unbounded_String (Boolean'Image (Read_Bool (FD)));
         when GGUF_TYPE_STR  => V_Str := To_Unbounded_String (Read_String (FD));
         when GGUF_TYPE_U64  => V_Str := To_Unbounded_String (U64'Image (Read_U64 (FD)));
         when GGUF_TYPE_I64  =>
            declare
               V : constant U64 := Read_U64 (FD);
            begin
               V_Str := To_Unbounded_String (U64'Image (V));
            end;
         when GGUF_TYPE_ARR =>
            declare
               Arr_Type : constant GGUF_Value_Type := GGUF_Value_Type'Val (Read_U32 (FD));
               Arr_Len  : constant U64 := Read_U64 (FD);
            begin
               V_Str := To_Unbounded_String ("[");
               for I in 1 .. Natural (Arr_Len) loop
                  if I > 1 then Append (V_Str, ", "); end if;
                  Append (V_Str, Read_Metadata_Value (FD, Arr_Type));
               end loop;
               Append (V_Str, "]");
            end;
      end case;
      return To_String (V_Str);
   end Read_Metadata_Value;

   --------------------------------------------------------------------
   -- GGML type mapping
   --------------------------------------------------------------------

   --  Map the on-disk GGML type code (which has gaps) to our enum. A naive
   --  'Val is wrong because the enum positions are contiguous (0..13) while
   --  the real codes skip values (Q5_0=6, …, Q5_K=13, Q6_K=14, Q8_K=15).
   function To_GGML_Type (V : U32) return GGML_Type is
   begin
      case V is
         when 0      => return GGML_TYPE_F32;
         when 1      => return GGML_TYPE_F16;
         when 2      => return GGML_TYPE_Q4_0;
         when 3      => return GGML_TYPE_Q4_1;
         when 6      => return GGML_TYPE_Q5_0;
         when 7      => return GGML_TYPE_Q5_1;
         when 8      => return GGML_TYPE_Q8_0;
         when 9      => return GGML_TYPE_Q8_1;
         when 10     => return GGML_TYPE_Q2_K;
         when 11     => return GGML_TYPE_Q3_K;
         when 12     => return GGML_TYPE_Q4_K;
         when 13     => return GGML_TYPE_Q5_K;
         when 14     => return GGML_TYPE_Q6_K;
         when 15     => return GGML_TYPE_Q8_K;
         when others => return GGML_TYPE_F32;  -- legacy/IQ types: unsupported
      end case;
   end To_GGML_Type;

   --------------------------------------------------------------------
   -- Open — parse the full GGUF header
   --------------------------------------------------------------------

   procedure Open (File : out GGUF_File; Path : String) is
      FD     : C_Int;
      Magic  : String (1 .. 4);
      Version : U32;
      N_Tensors : U64;
      N_Meta    : U64;
   begin
      -- Open file via POSIX
      FD := C_Open (Path, O_RDONLY, 0);
      if FD < 0 then
         Ada.Text_IO.Put_Line ("ERROR: cannot open " & Path);
         return;
      end if;

      -- Read magic "GGUF"
      Read_Exact (FD, Magic'Address, 4);
      if Magic /= "GGUF" then
         Ada.Text_IO.Put_Line ("ERROR: not a GGUF file");
         Ignore (C_Close (FD));
         return;
      end if;

      -- Version
      Version := Read_U32 (FD);
      if Version /= 2 and Version /= 3 then
         Ada.Text_IO.Put_Line ("GGUF version" & U32'Image (Version) & " (expected v2 or v3)");
      end if;

      -- Tensor count + metadata count
      N_Tensors := Read_U64 (FD);
      N_Meta    := Read_U64 (FD);

      File.Is_Open := True;
      File.Path := To_Unbounded_String (Path);
      File.Version := Version;
      File.FD := Integer (FD);

      Ada.Text_IO.Put_Line ("GGUF: version" & U32'Image (Version) &
        ", tensors:" & U64'Image (N_Tensors) &
        ", metadata:" & U64'Image (N_Meta));

      -- Parse metadata key-value pairs
      for I in 1 .. Natural (N_Meta) loop
         declare
            Key  : constant String := Read_String (FD);
            VT_Int : constant U32   := Read_U32 (FD);
            VT   : constant GGUF_Value_Type := GGUF_Value_Type'Val (Integer (VT_Int));
            M    : Metadata_Entry;
            Val  : Unbounded_String;
         begin
            if VT = GGUF_TYPE_ARR
              and then (Key = "tokenizer.ggml.tokens"
                        or else Key = "tokenizer.ggml.merges")
            then
               --  Capture string arrays element-by-element so token text
               --  containing commas/brackets survives intact.
               declare
                  Arr_Type : constant GGUF_Value_Type :=
                    GGUF_Value_Type'Val (Read_U32 (FD));
                  Arr_Len  : constant U64 := Read_U64 (FD);
               begin
                  for J in 1 .. Natural (Arr_Len) loop
                     declare
                        S : constant String := Read_Metadata_Value (FD, Arr_Type);
                     begin
                        if Key = "tokenizer.ggml.tokens" then
                           File.Tokens.Append (To_Unbounded_String (S));
                        else
                           File.Merges.Append (To_Unbounded_String (S));
                        end if;
                     end;
                  end loop;
                  Val := To_Unbounded_String
                    ("[array:" & Natural'Image (Natural (Arr_Len)) & " ]");
               end;
            else
               Val := To_Unbounded_String (Read_Metadata_Value (FD, VT));
            end if;

            M.Key   := To_Unbounded_String (Key);
            M.Value := Val;
            File.Meta.Append (M);

            -- Debug: print first few metadata entries
            if I <= 5 then
               Ada.Text_IO.Put_Line ("  meta: " & Key & " = " & To_String (Val) &
                 " (type=" & Integer'Image (Integer (VT_Int)) & ")");
            end if;

            -- Extract alignment from metadata
            if Key = "general.alignment" then
               File.Alignment_Val := U64'Value (To_String (Val));
            end if;
         end;
      end loop;

      -- Parse tensor info descriptors
      for I in 1 .. Natural (N_Tensors) loop
         declare
            Info : Tensor_Info;
            Name : constant String := Read_String (FD);
            ND   : constant U32 := Read_U32 (FD);
         begin
            Info.Name := To_Unbounded_String (Name);
            Info.N_Dims := ND;

            -- Read dimension array (GGUF stores n_dims followed by that many u64 values)
            for D in 1 .. Natural (ND) loop
               Info.Dims (D) := Read_U64 (FD);
            end loop;
            -- Zero remaining dims
            for D in Natural (ND) + 1 .. 4 loop
               Info.Dims (D) := 0;
            end loop;

            -- Read GGML type + offset
            Info.Kind := To_GGML_Type (Read_U32 (FD));
            Info.Offset := Read_U64 (FD);

            File.Tensors.Append (Info);
         end;
      end loop;

      -- Tensor data begins at the next alignment boundary after the tensor
      -- info section. Per-tensor Info.Offset values are relative to this point.
      declare
         Pos : constant U64 := U64 (C_LSeek (FD, 0, SEEK_CUR));
         A   : constant U64 := File.Alignment_Val;
      begin
         File.Data_Start := ((Pos + A - 1) / A) * A;
      end;

      Ada.Text_IO.Put_Line ("GGUF: parsed" & Integer'Image (Natural (N_Tensors)) &
        " tensors, alignment=" & U64'Image (File.Alignment_Val));
   end Open;

   --------------------------------------------------------------------
   -- Accessors
   --------------------------------------------------------------------

   function Is_Open (File : GGUF_File) return Boolean is
   begin
      return File.Is_Open;
   end Is_Open;

   function Tensor_Count (File : GGUF_File) return Natural is
   begin
      return Natural (File.Tensors.Length);
   end Tensor_Count;

   function Metadata_Count (File : GGUF_File) return Natural is
   begin
      return Natural (File.Meta.Length);
   end Metadata_Count;

   function Metadata (File : GGUF_File; Key : String) return String is
   begin
      for M of File.Meta loop
         if To_String (M.Key) = Key then
            return To_String (M.Value);
         end if;
      end loop;
      return "";
   end Metadata;

   function Meta_Key_At (File : GGUF_File; Index : Positive) return String is
   begin
      return To_String (File.Meta (Index).Key);
   end Meta_Key_At;

   function Meta_Value_At (File : GGUF_File; Index : Positive) return String is
   begin
      return To_String (File.Meta (Index).Value);
   end Meta_Value_At;

   function Token_Count (File : GGUF_File) return Natural is
   begin
      return Natural (File.Tokens.Length);
   end Token_Count;

   function Token_At (File : GGUF_File; Index : Positive) return String is
   begin
      return To_String (File.Tokens (Index));
   end Token_At;

   function Merge_Count (File : GGUF_File) return Natural is
   begin
      return Natural (File.Merges.Length);
   end Merge_Count;

   function Merge_At (File : GGUF_File; Index : Positive) return String is
   begin
      return To_String (File.Merges (Index));
   end Merge_At;

   function Tensor_At (File : GGUF_File; Index : Positive) return Tensor_Info is
   begin
      return File.Tensors (Index);
   end Tensor_At;

   function Find_Tensor (File : GGUF_File; Name : String) return Tensor_Info is
   begin
      for I in 1 .. Natural (File.Tensors.Length) loop
         if To_String (File.Tensors (I).Name) = Name then
            return File.Tensors (I);
         end if;
      end loop;
      raise Constraint_Error with "Tensor not found: " & Name;
   end Find_Tensor;

   procedure Read_Tensor_Raw
     (File   : in out GGUF_File;
      Info   : Tensor_Info;
      Buffer : System.Address;
      Buf_Size : Natural)
   is
      FD : constant C_Int := C_Int (File.FD);
      --  Info.Offset is relative to the aligned data-section start.
      Off : constant C_Off := C_Off (File.Data_Start + Info.Offset);
   begin
      Ignore (C_LSeek (FD, Off, SEEK_SET));
      Read_Exact (FD, Buffer, Buf_Size);
   end Read_Tensor_Raw;

   function Tensor_Byte_Size (Info : Tensor_Info) return U64 is
      N_Elements : constant U64 := Tensor_Num_Elements (Info);
   begin
      case Info.Kind is
         when GGML_TYPE_F32 => return N_Elements * 4;
         when GGML_TYPE_F16 => return N_Elements * 2;
         when GGML_TYPE_Q5_0 => return ((N_Elements + 31) / 32) * 64;
         when GGML_TYPE_Q5_1 => return ((N_Elements + 31) / 32) * 64;
         --  K-quant super-block sizes (bytes per 256 elements), per llama.cpp:
         when GGML_TYPE_Q4_K => return (N_Elements / 256) * 144;
         when GGML_TYPE_Q5_K => return (N_Elements / 256) * 176;
         when GGML_TYPE_Q6_K => return (N_Elements / 256) * 210;
         when GGML_TYPE_Q8_K => return (N_Elements / 256) * 292;
         when GGML_TYPE_Q8_0 => return N_Elements;
         when others => return N_Elements * 4;
      end case;
   end Tensor_Byte_Size;

   function Tensor_Num_Elements (Info : Tensor_Info) return U64 is
      N : U64 := 1;
   begin
      for D in 1 .. Natural (Info.N_Dims) loop
         N := N * Info.Dims (D);
      end loop;
      return N;
   end Tensor_Num_Elements;

   function Alignment (File : GGUF_File) return U64 is
   begin
      return File.Alignment_Val;
   end Alignment;

   procedure Close (File : in out GGUF_File) is
   begin
      if File.FD >= 0 then
         Ignore (C_Close (C_Int (File.FD)));
         File.FD := -1;
      end if;
      File.Is_Open := False;
      File.Tensors.Clear;
      File.Meta.Clear;
   end Close;

   function Is_GGUF (Path : String) return Boolean is
      FD : C_Int;
      Magic : String (1 .. 4);
   begin
      FD := C_Open (Path, O_RDONLY, 0);
      if FD < 0 then
         return False;
      end if;
      declare
         N : C_Int;
      begin
         N := C_Read (FD, Magic'Address, 4);
         if N /= 4 then
            Ignore (C_Close (FD));
            return False;
         end if;
      end;
      Ignore (C_Close (FD));
      return Magic = "GGUF";
   end Is_GGUF;

end LLM_GGUF;
