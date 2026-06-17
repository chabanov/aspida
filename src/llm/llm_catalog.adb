---------------------------------------------------------------------
-- LLM_Catalog body — directory walk + metadata-only classification.
---------------------------------------------------------------------

with Ada.Directories;            use Ada.Directories;
with Ada.Environment_Variables;
with Ada.Characters.Handling;    use Ada.Characters.Handling;
with Ada.Strings.Fixed;
with LLM_Engine;
with LLM_GGUF;

package body LLM_Catalog is

   use SU;

   Max_Depth : constant := 6;   -- bound the walk against symlink loops

   --------------------------------------------------------------------
   --  Search roots
   --------------------------------------------------------------------

   package Str_Vectors is new Ada.Containers.Vectors (Positive, Unbounded_String);

   function Home return String is
     (if Ada.Environment_Variables.Exists ("HOME")
      then Ada.Environment_Variables.Value ("HOME") else "");

   --  Build the ordered, existing, de-duplicated list of directories to scan.
   function Roots return Str_Vectors.Vector is
      V : Str_Vectors.Vector;

      procedure Add (Dir : String) is
      begin
         if Dir'Length = 0 then
            return;
         end if;
         for Existing of V loop          -- de-dup
            if To_String (Existing) = Dir then
               return;
            end if;
         end loop;
         if Exists (Dir) and then Kind (Dir) = Directory then
            V.Append (To_Unbounded_String (Dir));
         end if;
      exception
         when others => null;            -- unreadable path: skip
      end Add;

      --  ASPIDA_MODELS_DIR: ':'-separated list, highest precedence.
      procedure Add_Env_List is
         E    : constant String :=
           (if Ada.Environment_Variables.Exists ("ASPIDA_MODELS_DIR")
            then Ada.Environment_Variables.Value ("ASPIDA_MODELS_DIR") else "");
         From : Positive := E'First;
      begin
         for I in E'Range loop
            if E (I) = ':' then
               Add (E (From .. I - 1));
               From := I + 1;
            end if;
         end loop;
         if From <= E'Last then
            Add (E (From .. E'Last));
         end if;
      end Add_Env_List;

      H : constant String := Home;
   begin
      Add_Env_List;
      Add ("models");                                  -- repo-local ./models
      if H'Length > 0 then
         Add (H & "/.lmstudio/models");
         Add (H & "/.cache/lm-studio/models");
         Add (H & "/.cache/huggingface");
         Add (H & "/.ollama/models");
         Add (H & "/models");
      end if;
      Add ("/root/models");
      Add ("/root/aspida/models");
      return V;
   end Roots;

   --------------------------------------------------------------------
   --  Classification helpers
   --------------------------------------------------------------------

   function Has_GGUF_Ext (Name : String) return Boolean is
     (Name'Length > 5
      and then To_Lower (Name (Name'Last - 4 .. Name'Last)) = ".gguf");

   function Is_Projector (Base : String) return Boolean is
     (Base'Length >= 6 and then To_Lower (Base (Base'First .. Base'First + 5)) = "mmproj");

   --  Pull a quant tag (Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, F16, …) out of the
   --  file name. Returns "" if none recognised.
   function Parse_Quant (Base : String) return String is
      Up    : constant String := To_Upper (Base);
      Cand  : constant array (Positive range <>) of Unbounded_String :=
        [To_Unbounded_String ("Q2_K"),   To_Unbounded_String ("Q3_K_S"),
         To_Unbounded_String ("Q3_K_M"), To_Unbounded_String ("Q3_K_L"),
         To_Unbounded_String ("Q4_K_S"), To_Unbounded_String ("Q4_K_M"),
         To_Unbounded_String ("Q5_K_S"), To_Unbounded_String ("Q5_K_M"),
         To_Unbounded_String ("Q6_K"),   To_Unbounded_String ("Q8_K_P"),
         To_Unbounded_String ("Q8_K"),   To_Unbounded_String ("Q4_0"),
         To_Unbounded_String ("Q5_0"),   To_Unbounded_String ("Q8_0"),
         To_Unbounded_String ("BF16"),   To_Unbounded_String ("F16"),
         To_Unbounded_String ("F32")];
      Best  : Natural := 0;      -- prefer the longest match (Q4_K_M over Q4_K)
      Hit   : Unbounded_String;
   begin
      for C of Cand loop
         declare
            S : constant String := To_String (C);
         begin
            if Ada.Strings.Fixed.Index (Up, S) /= 0 and then S'Length > Best then
               Best := S'Length;
               Hit  := C;
            end if;
         end;
      end loop;
      return To_String (Hit);
   end Parse_Quant;

   --  Open once, read the three metadata keys we display, then close.
   procedure Probe_Meta
     (Path : String; Arch, Name, Params : out Unbounded_String)
   is
      G : LLM_GGUF.GGUF_File;
   begin
      Arch := Null_Unbounded_String;
      Name := Null_Unbounded_String;
      Params := Null_Unbounded_String;
      LLM_GGUF.Open (G, Path);
      if LLM_GGUF.Is_Open (G) then
         Arch   := To_Unbounded_String (LLM_GGUF.Metadata (G, "general.architecture"));
         Name   := To_Unbounded_String (LLM_GGUF.Metadata (G, "general.name"));
         Params := To_Unbounded_String (LLM_GGUF.Metadata (G, "general.size_label"));
         LLM_GGUF.Close (G);
      end if;
   exception
      when others =>
         begin
            LLM_GGUF.Close (G);
         exception
            when others => null;
         end;
   end Probe_Meta;

   function Classify (Path : String) return Model_Entry is
      Base : constant String := Simple_Name (Path);
      E    : Model_Entry;
   begin
      E.Path := To_Unbounded_String (Path);
      begin
         E.Size := Long_Long_Integer (Ada.Directories.Size (Path));
      exception
         when others => E.Size := 0;
      end;
      E.Quant := To_Unbounded_String (Parse_Quant (Base));

      if Is_Projector (Base) then
         E.Name := To_Unbounded_String (Base);
         E.Status := Projector;
         return E;
      end if;

      Probe_Meta (Path, E.Arch, E.Name, E.Params);
      if Length (E.Name) = 0 then
         E.Name := To_Unbounded_String (Base);   -- fall back to the file name
      end if;

      if Length (E.Arch) = 0 then
         E.Status := Invalid;
      elsif LLM_Engine.Supports (To_String (E.Arch)) then
         E.Status := Supported;
      else
         E.Status := Unsupported;
      end if;
      return E;
   end Classify;

   --------------------------------------------------------------------
   --  Walk
   --------------------------------------------------------------------

   procedure Walk
     (Dir : String; Depth : Natural; Acc : in out Entry_Vectors.Vector)
   is
      S    : Search_Type;
      Item : Directory_Entry_Type;
   begin
      Start_Search
        (S, Dir, "",
         Filter => [Ordinary_File => True, Directory => True,
                    Special_File => False]);
      while More_Entries (S) loop
         Get_Next_Entry (S, Item);
         declare
            Base : constant String := Simple_Name (Item);
            Full : constant String := Full_Name (Item);
         begin
            if Kind (Item) = Directory then
               if Base /= "." and then Base /= ".." and then Depth < Max_Depth then
                  Walk (Full, Depth + 1, Acc);
               end if;
            elsif Kind (Item) = Ordinary_File and then Has_GGUF_Ext (Base) then
               Acc.Append (Classify (Full));
            end if;
         exception
            when others => null;          -- skip a bad entry, keep scanning
         end;
      end loop;
      End_Search (S);
   exception
      when others => null;                -- unreadable directory: skip
   end Walk;

   --------------------------------------------------------------------
   --  Ordering
   --------------------------------------------------------------------

   function Rank (S : Model_Status) return Natural is
     (case S is when Supported => 0, when Unsupported => 1,
                when Projector => 2, when Invalid => 3);

   function Less (L, R : Model_Entry) return Boolean is
   begin
      if Rank (L.Status) /= Rank (R.Status) then
         return Rank (L.Status) < Rank (R.Status);
      end if;
      if To_Lower (To_String (L.Name)) /= To_Lower (To_String (R.Name)) then
         return To_Lower (To_String (L.Name)) < To_Lower (To_String (R.Name));
      end if;
      return To_String (L.Path) < To_String (R.Path);
   end Less;

   package Sorter is new Entry_Vectors.Generic_Sorting ("<" => Less);

   --------------------------------------------------------------------
   --  Public API
   --------------------------------------------------------------------

   function Discover return Entry_Vectors.Vector is
      Result : Entry_Vectors.Vector;

      function Already_Have (Path : String) return Boolean is
      begin
         for E of Result loop
            if To_String (E.Path) = Path then
               return True;
            end if;
         end loop;
         return False;
      end Already_Have;

      Found : Entry_Vectors.Vector;
   begin
      for R of Roots loop
         Walk (To_String (R), 0, Found);
      end loop;
      for E of Found loop                 -- de-dup by absolute path
         if not Already_Have (To_String (E.Path)) then
            Result.Append (E);
         end if;
      end loop;
      Sorter.Sort (Result);
      return Result;
   end Discover;

   function Roots_Description return String is
      First : Boolean := True;
      Acc   : Unbounded_String;
   begin
      for R of Roots loop
         if not First then
            Append (Acc, ":");
         end if;
         Append (Acc, R);
         First := False;
      end loop;
      return To_String (Acc);
   end Roots_Description;

   function Human_Size (Bytes : Long_Long_Integer) return String is
      GiB : constant Long_Long_Integer := 1024 ** 3;
      MiB : constant Long_Long_Integer := 1024 ** 2;

      function One_DP (Num, Den : Long_Long_Integer; Unit : String) return String is
         Whole : constant Long_Long_Integer := Num / Den;
         Tenth : constant Long_Long_Integer := (Num * 10 / Den) mod 10;
      begin
         return Ada.Strings.Fixed.Trim (Whole'Image, Ada.Strings.Left)
           & "." & Ada.Strings.Fixed.Trim (Tenth'Image, Ada.Strings.Left)
           & " " & Unit;
      end One_DP;
   begin
      if Bytes >= GiB then
         return One_DP (Bytes, GiB, "GB");
      elsif Bytes >= MiB then
         return Ada.Strings.Fixed.Trim
                  (Long_Long_Integer'Image (Bytes / MiB), Ada.Strings.Left) & " MB";
      else
         return Ada.Strings.Fixed.Trim (Bytes'Image, Ada.Strings.Left) & " B";
      end if;
   end Human_Size;

   function Describe (E : Model_Entry) return String is
      Tag : constant String :=
        (case E.Status is
           when Supported   => "[ok]   ",
           when Unsupported => "[arch] ",
           when Projector   => "[proj] ",
           when Invalid     => "[bad]  ");
      Arch_S  : constant String :=
        (if Length (E.Arch) > 0 then To_String (E.Arch) else "?");
      Quant_S : constant String :=
        (if Length (E.Quant) > 0 then " " & To_String (E.Quant) else "");
      Par_S   : constant String :=
        (if Length (E.Params) > 0 then " " & To_String (E.Params) else "");
   begin
      return Tag & To_String (E.Name)
        & "  (" & Arch_S & Par_S & Quant_S & ", " & Human_Size (E.Size) & ")"
        & ASCII.LF & "         " & To_String (E.Path);
   end Describe;

end LLM_Catalog;
