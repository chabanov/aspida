---------------------------------------------------------------------
-- LLM_GGUF body — GGUF v3 parser, now source-agnostic
--
-- All byte I/O (sequential header parse + random tensor reads) is routed
-- through a LLM_Byte_Source.Byte_Source. Today that is always a
-- Local_File_Source (POSIX fd + lseek + read, implemented in LLM_Byte_Source);
-- H19 will swap in a Remote_AEAD_Source with no change here. The POSIX thin
-- bindings and the short-read loop moved to llm_byte_source.adb verbatim.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Exceptions;
--  Interfaces and LLM_Byte_Source are withed by the spec, so they are visible
--  in the body without a redundant with clause here.

package body LLM_GGUF is

   use Ada.Strings.Unbounded;
   use LLM_Byte_Source;  --  makes Byte_Source_Access's predefined = directly visible

   --  Shorthand for the byte-source access the file holds.
   subtype Source_Access is LLM_Byte_Source.Byte_Source_Access;

   --  Sanity caps on GGUF metadata sizes. A hostile or truncated file could
   --  otherwise advertise a multi-gigabyte string / millions of tensors and
   --  either OOM the loader or pin it in a long read loop (DoS). Real GGUFs
   --  stay well under these (vocab ≤ 256k, tensors ≤ ~1k, merges ≤ ~150k).
   Max_String_Len   : constant U64 := 1_048_576;   -- 1 MiB per metadata string
   Max_Meta_Count   : constant U64 := 16_384;      -- key/value pairs
   Max_Tensor_Count : constant U64 := 16_384;      -- tensor descriptors
   Max_Array_Len    : constant U64 := 1_048_576;   -- metadata array elements

   --------------------------------------------------------------------
   -- Binary reader helpers (all routed through the Byte_Source cursor)
   --------------------------------------------------------------------

   --  Read exactly Count bytes at the source's cursor, advancing it. The
   --  source's Malformed_Source (short read) is translated to the
   --  Constraint_Error the parser historically raised, so Open's exception
   --  handler wraps it into Malformed_GGUF unchanged.
   procedure Read_Exact (S : Source_Access; Addr : System.Address; Count : Natural) is
   begin
      S.Read_Seq (Addr, Count);
   exception
      when Malformed_Source =>
         raise Constraint_Error with "Short read from GGUF file";
   end Read_Exact;

   type U8_T  is mod 2 ** 8;
   function Read_U8 (S : Source_Access) return U8_T is
      V : U8_T := 0;
   begin
      Read_Exact (S, V'Address, 1);
      return V;
   end Read_U8;

   type U16_T is mod 2 ** 16;
   function Read_U16 (S : Source_Access) return U16_T is
      V : U16_T := 0;
   begin
      Read_Exact (S, V'Address, 2);
      return V;
   end Read_U16;

   function Read_U32 (S : Source_Access) return U32 is
      V : U32 := 0;
   begin
      Read_Exact (S, V'Address, 4);
      return V;
   end Read_U32;

   function Read_U64 (S : Source_Access) return U64 is
      V : U64 := 0;
   begin
      Read_Exact (S, V'Address, 8);
      return V;
   end Read_U64;

   function Read_F32 (S : Source_Access) return Float is
      V : Float := 0.0;
   begin
      Read_Exact (S, V'Address, 4);
      return V;
   end Read_F32;

   function Read_Bool (S : Source_Access) return Boolean is
   begin
      --  GGUF BOOL is a single byte (not a 32-bit word) — reading 4 would
      --  desync the cursor and corrupt every later metadata entry.
      return Read_U8 (S) /= 0;
   end Read_Bool;

   function Read_String (S : Source_Access) return String is
      Len : constant U64 := Read_U64 (S);
   begin
      if Len > Max_String_Len then
         raise Constraint_Error with "GGUF string length out of range";
      end if;
      declare
         S2 : String (1 .. Natural (Len));
      begin
         if Len > 0 then
            Read_Exact (S, S2'Address, Natural (Len));
         end if;
         return S2;
      end;
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

   --  Convert an on-disk value-type code to the enum, rejecting any code that
   --  is out of range BEFORE 'Val (which would raise a raw Constraint_Error on
   --  a hostile code >= 12). The code is attacker-controlled.
   function To_Value_Type (Code : U32) return GGUF_Value_Type is
   begin
      if Code > U32 (GGUF_Value_Type'Pos (GGUF_Value_Type'Last)) then
         raise Malformed_GGUF
           with "GGUF metadata value-type code out of range:" & U32'Image (Code);
      end if;
      return GGUF_Value_Type'Val (Code);
   end To_Value_Type;

   function Read_Metadata_Value (S : Source_Access; VT : GGUF_Value_Type) return String is
      V_Str : Unbounded_String;
   begin
      case VT is
         --  Small scalars are 1 or 2 bytes on disk; reading 4 would desync the
         --  file cursor and corrupt every subsequent metadata entry.
         when GGUF_TYPE_U8   => V_Str := To_Unbounded_String (U64'Image (U64 (Read_U8 (S))));
         when GGUF_TYPE_I8   =>
            declare
               B : constant U8_T := Read_U8 (S);
            begin
               V_Str := To_Unbounded_String (Integer'Image
                 (if B >= 128 then Integer (B) - 256 else Integer (B)));
            end;
         when GGUF_TYPE_U16  => V_Str := To_Unbounded_String (U64'Image (U64 (Read_U16 (S))));
         when GGUF_TYPE_I16  =>
            declare
               H : constant U16_T := Read_U16 (S);
            begin
               V_Str := To_Unbounded_String (Integer'Image
                 (if H >= 32768 then Integer (H) - 65536 else Integer (H)));
            end;
         when GGUF_TYPE_U32  => V_Str := To_Unbounded_String (U64'Image (U64 (Read_U32 (S))));
         when GGUF_TYPE_I32  => V_Str := To_Unbounded_String (Integer'Image (Integer (Read_U32 (S))));
         when GGUF_TYPE_F32  => V_Str := To_Unbounded_String (Float'Image (Read_F32 (S)));
         when GGUF_TYPE_BOOL => V_Str := To_Unbounded_String (Boolean'Image (Read_Bool (S)));
         when GGUF_TYPE_STR  => V_Str := To_Unbounded_String (Read_String (S));
         when GGUF_TYPE_U64  => V_Str := To_Unbounded_String (U64'Image (Read_U64 (S)));
         when GGUF_TYPE_I64  =>
            declare
               V : constant U64 := Read_U64 (S);
            begin
               V_Str := To_Unbounded_String (U64'Image (V));
            end;
         when GGUF_TYPE_ARR =>
            declare
               Arr_Type : constant GGUF_Value_Type := To_Value_Type (Read_U32 (S));
               Arr_Len  : constant U64 := Read_U64 (S);
            begin
               if Arr_Len > Max_Array_Len then
                  raise Constraint_Error with "GGUF array length out of range";
               end if;
               V_Str := To_Unbounded_String ("[");
               for I in 1 .. Natural (Arr_Len) loop
                  if I > 1 then Append (V_Str, ", "); end if;
                  Append (V_Str, Read_Metadata_Value (S, Arr_Type));
               end loop;
               Append (V_Str, "]");
            end;
      end case;
      return To_String (V_Str);
   end Read_Metadata_Value;

   --------------------------------------------------------------------
   -- Checked tensor size math (single source of truth)
   --------------------------------------------------------------------

   --  All tensor element-count / byte-size arithmetic flows through here so it
   --  is computed exactly ONCE, with overflow checks ENABLED, on the untrusted
   --  dims + type. Three call sites used to do this three divergent ways
   --  (U64 modular wrap, suppressed Long_Long_Integer, suppressed 32-bit Int);
   --  any disagreement fed an allocation, the decode loop count, and the read
   --  size with no cross-check -> OOB read/write. Now the validated byte size is
   --  stored in Tensor_Info.Byte_Size and everyone consumes that.
   --
   --  A weight that genuinely needs > ~2^63 bytes is not a real model; we reject
   --  it as malformed rather than wrapping silently.
   Max_Tensor_Bytes : constant U64 := 2 ** 62;  -- generous, well below 2^63

   --  Multiply two U64s, raising Malformed_GGUF on overflow past Max_Tensor_Bytes.
   function Checked_Mul (A, B : U64) return U64 is
   begin
      if A /= 0 and then B > Max_Tensor_Bytes / A then
         raise Malformed_GGUF with "GGUF tensor size overflow";
      end if;
      return A * B;
   end Checked_Mul;

   --  Add two U64s, raising Malformed_GGUF on overflow past Max_Tensor_Bytes.
   function Checked_Add (A, B : U64) return U64 is
   begin
      if A > Max_Tensor_Bytes - B then
         raise Malformed_GGUF with "GGUF tensor size overflow";
      end if;
      return A + B;
   end Checked_Add;

   --  Element count = product of the declared dims, overflow-checked.
   function Checked_Num_Elements (Info : Tensor_Info) return U64 is
      N : U64 := 1;
   begin
      for D in 1 .. Natural (Info.N_Dims) loop
         N := Checked_Mul (N, Info.Dims (D));
      end loop;
      return N;
   end Checked_Num_Elements;

   --  Byte size from element count + type, overflow-checked. This is the only
   --  place the per-type block math lives.
   function Checked_Byte_Size (Info : Tensor_Info) return U64 is
      N : constant U64 := Checked_Num_Elements (Info);
   begin
      case Info.Kind is
         when GGML_TYPE_F32  => return Checked_Mul (N, 4);
         when GGML_TYPE_F16  => return Checked_Mul (N, 2);
         when GGML_TYPE_BF16 => return Checked_Mul (N, 2);
         when GGML_TYPE_Q4_0 => return Checked_Mul ((N + 31) / 32, 18);
         when GGML_TYPE_Q5_0 => return Checked_Mul ((N + 31) / 32, 22);
         when GGML_TYPE_Q5_1 => return Checked_Mul ((N + 31) / 32, 24);
         when GGML_TYPE_Q2_K => return Checked_Mul ((N + 255) / 256, 84);
         when GGML_TYPE_Q3_K => return Checked_Mul ((N + 255) / 256, 110);
         when GGML_TYPE_Q4_K => return Checked_Mul ((N + 255) / 256, 144);
         when GGML_TYPE_Q5_K => return Checked_Mul ((N + 255) / 256, 176);
         when GGML_TYPE_Q6_K => return Checked_Mul ((N + 255) / 256, 210);
         when GGML_TYPE_Q8_K => return Checked_Mul ((N + 255) / 256, 292);
         when GGML_TYPE_Q8_0 => return Checked_Mul ((N + 31) / 32, 34);
         when others         => return Checked_Mul (N, 4);
      end case;
   end Checked_Byte_Size;

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
         when 30     => return GGML_TYPE_BF16;
         when others => return GGML_TYPE_UNKNOWN;
           --  IQ* / ternary / legacy types are NOT F32. Mapping them to F32
           --  used to defeat Is_Supported (which then returned True), so IQ
           --  tensors loaded and produced garbage at inference. UNKNOWN keeps
           --  them out of every decode path; Is_Supported returns False and
           --  the loaders reject the file with a clear message.
      end case;
   end To_GGML_Type;

   --------------------------------------------------------------------
   -- Open — parse the full GGUF header
   --------------------------------------------------------------------

   procedure Open (File : out GGUF_File; Path : String) is
      Src : Source_Access;
   begin
      --  Open the byte source. Open_Source returns null (rather than raising)
      --  on a missing / unreadable file, mirroring the historical semantics;
      --  an actual read error during the parse below raises via Read_Exact.
      Src := LLM_Byte_Source.Open_Source (Path);
      if Src = null then
         Ada.Text_IO.Put_Line ("ERROR: cannot open " & Path);
         return;
      end if;
      --  Hand ownership to Open_From_Source, which parses the header and
      --  stores the source in File (freed on Close / on a parse failure).
      Open_From_Source (File, Src);
   end Open;

   --------------------------------------------------------------------
   -- Open_From_Source — parse the GGUF header from an already-open source
   --------------------------------------------------------------------

   procedure Open_From_Source (File : out GGUF_File; Src : Source_Access) is
      Source     : Source_Access renames Src;  --  keep the parse body readable
      Magic      : String (1 .. 4);
      Version    : U32;
      N_Tensors  : U64;
      N_Meta     : U64;
   begin
      --  Take ownership of the source up front so EVERY failure path (a bad
      --  magic, an absurd count, or a mid-parse exception) frees it via
      --  Close (File). This also fixes a latent leak in the old Open, where a
      --  short read before File.Source was assigned would orphan the source.
      File.Source := Source;

      --  Parse from the start of the source regardless of where its cursor
      --  currently sits. A caller may have consumed the source before handing
      --  it over — e.g. LLM_Weight_Pin.Hash_Source reads the whole thing to
      --  verify a pinned digest and leaves the cursor at EOF — and the magic
      --  read below must not pick up from that stale position.
      Source.Seek (0);

      -- Read magic "GGUF"
      Read_Exact (Source, Magic'Address, 4);
      if Magic /= "GGUF" then
         Ada.Text_IO.Put_Line ("ERROR: not a GGUF file");
         Close (File);
         return;
      end if;

      -- Version
      Version := Read_U32 (Source);
      if Version /= 2 and Version /= 3 then
         Ada.Text_IO.Put_Line ("GGUF version" & U32'Image (Version) & " (expected v2 or v3)");
      end if;

      -- Tensor count + metadata count
      N_Tensors := Read_U64 (Source);
      N_Meta    := Read_U64 (Source);

      --  Reject an absurd count before allocating / looping. A hostile header
      --  could otherwise force a multi-million-iteration parse (DoS).
      if N_Tensors > Max_Tensor_Count or else N_Meta > Max_Meta_Count then
         Ada.Text_IO.Put_Line
           ("ERROR: GGUF header reports" & U64'Image (N_Tensors)
            & " tensors /" & U64'Image (N_Meta)
            & " metadata (exceeds sanity cap)");
         Close (File);
         return;
      end if;

      File.Is_Open := True;
      File.Version := Version;
      --  File.Source was assigned at the top; the path is unknown for a
      --  remote source, so File.Path stays empty (it is only a debug label).

      Ada.Text_IO.Put_Line ("GGUF: version" & U32'Image (Version) &
        ", tensors:" & U64'Image (N_Tensors) &
        ", metadata:" & U64'Image (N_Meta));

      -- Parse metadata key-value pairs
      for I in 1 .. Natural (N_Meta) loop
         declare
            Key  : constant String := Read_String (Source);
            VT_Int : constant U32   := Read_U32 (Source);
            VT   : constant GGUF_Value_Type := To_Value_Type (VT_Int);
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
                    To_Value_Type (Read_U32 (Source));
                  Arr_Len  : constant U64 := Read_U64 (Source);
               begin
                  if Arr_Len > Max_Array_Len then
                     raise Constraint_Error
                       with "tokenizer array length out of range";
                  end if;
                  for J in 1 .. Natural (Arr_Len) loop
                     declare
                        S : constant String := Read_Metadata_Value (Source, Arr_Type);
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
               Val := To_Unbounded_String (Read_Metadata_Value (Source, VT));
            end if;

            M.Key   := To_Unbounded_String (Key);
            M.Value := Val;
            File.Meta.Append (M);

            -- Debug: print first few metadata entries
            if I <= 5 then
               Ada.Text_IO.Put_Line ("  meta: " & Key & " = " & To_String (Val) &
                 " (type=" & Integer'Image (Integer (VT_Int)) & ")");
            end if;

            -- Extract alignment from metadata. The value comes from an
            -- attacker-controlled string and is later used as a divisor for the
            -- data-start rounding, so a non-numeric / zero / absurd value would
            -- raise (U64'Value) or divide by zero. Validate: must parse, be a
            -- power of two, and lie in 1 .. 65536; otherwise keep the default 32.
            if Key = "general.alignment" then
               declare
                  A : U64;
               begin
                  A := U64'Value (To_String (Val));
                  if A >= 1 and then A <= 65_536
                    and then (A and (A - 1)) = 0   -- power of two
                  then
                     File.Alignment_Val := A;
                  else
                     Ada.Text_IO.Put_Line
                       ("WARN: ignoring invalid general.alignment="
                        & To_String (Val) & " (keeping 32)");
                  end if;
               exception
                  when others =>
                     Ada.Text_IO.Put_Line
                       ("WARN: unparsable general.alignment="
                        & To_String (Val) & " (keeping 32)");
               end;
            end if;
         end;
      end loop;

      -- Parse tensor info descriptors
      for I in 1 .. Natural (N_Tensors) loop
         declare
            Info : Tensor_Info;
            Name : constant String := Read_String (Source);
            ND   : constant U32 := Read_U32 (Source);
         begin
            --  Dims is a 4-element array; a hostile ND > 4 would index out of
            --  bounds in the dim loop below. Reject before reading any dims.
            if ND > 4 then
               raise Malformed_GGUF
                 with "GGUF tensor """ & Name & """ has" & U32'Image (ND)
                      & " dims (max 4)";
            end if;

            Info.Name := To_Unbounded_String (Name);
            Info.N_Dims := ND;

            -- Read dimension array (GGUF stores n_dims followed by that many u64 values)
            for D in 1 .. Natural (ND) loop
               Info.Dims (D) := Read_U64 (Source);
            end loop;
            -- Zero remaining dims
            for D in Natural (ND) + 1 .. 4 loop
               Info.Dims (D) := 0;
            end loop;

            -- Read GGML type + offset
            Info.Kind := To_GGML_Type (Read_U32 (Source));
            Info.Offset := Read_U64 (Source);

            --  Compute the validated, overflow-checked byte size ONCE here and
            --  store it; this is the single source of truth consumed by every
            --  allocation, the decode loop count, and the read size. (File-end
            --  bounds are checked in a second pass below, once Data_Start and
            --  File_Size are known.)
            Info.Byte_Size := Checked_Byte_Size (Info);

            File.Tensors.Append (Info);
         end;
      end loop;

      -- Tensor data begins at the next alignment boundary after the tensor
      -- info section. Per-tensor Info.Offset values are relative to this point.
      declare
         Pos : constant U64 := U64 (Source.Cursor);
         A   : constant U64 := File.Alignment_Val;
      begin
         File.Data_Start := ((Pos + A - 1) / A) * A;
      end;

      --  Determine the real source length once (the Byte_Source probed it at
      --  Open; querying it does not move the cursor). Used to validate that
      --  every tensor's [Data_Start + Offset, + Byte_Size) range actually lies
      --  within the source — a hostile descriptor could otherwise advertise an
      --  offset/size that reads past the end (OOB).
      File.File_Size := U64 (Source.Byte_Length);

      --  Restore the cursor to the aligned data start, so a sequential read
      --  after the header (if any) begins there. Random tensor reads below do
      --  their own Seek, so this is for parity only — and MUST stay tolerant:
      --  a valid metadata-only / zero-tensor GGUF (or one written without
      --  trailing alignment padding) has an aligned Data_Start that rounds up
      --  past the physical end, and a strict Seek would reject it. Swallow the
      --  past-EOF case (the old lseek-and-discard behaviour); a genuinely
      --  short file is still caught by the per-tensor End_Off > File_Size
      --  bound below.
      begin
         Source.Seek (Interfaces.Unsigned_64 (File.Data_Start));
      exception
         when LLM_Byte_Source.Malformed_Source =>
            null;
      end;

      --  Validate every tensor against the source length using checked 64-bit
      --  arithmetic (Checked_Add rejects overflow in the addition itself).
      for I in 1 .. Natural (File.Tensors.Length) loop
         declare
            T       : constant Tensor_Info := File.Tensors (I);
            Abs_Off : constant U64 := Checked_Add (File.Data_Start, T.Offset);
            End_Off : constant U64 := Checked_Add (Abs_Off, T.Byte_Size);
         begin
            if End_Off > File.File_Size then
               raise Malformed_GGUF
                 with "GGUF tensor """ & To_String (T.Name)
                      & """ extends past end of file (needs"
                      & U64'Image (End_Off) & " bytes, file is"
                      & U64'Image (File.File_Size) & ")";
            end if;
         end;
      end loop;

      Ada.Text_IO.Put_Line ("GGUF: parsed" & Integer'Image (Natural (N_Tensors)) &
        " tensors, alignment=" & U64'Image (File.Alignment_Val));
   exception
      --  Any failure mid-parse (malformed header, short read, overflow) must
      --  not leak the byte source (owned by File since the top of this body)
      --  and must not surface as a raw Constraint_Error. Close+free the source
      --  and re-raise as Malformed_GGUF.
      when Malformed_GGUF =>
         Close (File);
         raise;
      when E : others =>
         Close (File);
         raise Malformed_GGUF
           with "GGUF parse error: " & Ada.Exceptions.Exception_Message (E);
   end Open_From_Source;

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

   function Has_Tensor (File : GGUF_File; Name : String) return Boolean is
   begin
      for I in 1 .. Natural (File.Tensors.Length) loop
         if To_String (File.Tensors (I).Name) = Name then
            return True;
         end if;
      end loop;
      return False;
   end Has_Tensor;

   procedure Read_Tensor_Raw
     (File   : in out GGUF_File;
      Info   : Tensor_Info;
      Buffer : System.Address;
      Buf_Size : Natural)
   is
      --  Info.Offset is relative to the aligned data-section start. Use checked
      --  64-bit arithmetic and bound the read against the source length so a
      --  malformed tensor (validated in Open, but re-checked here defensively)
      --  can never read past the end. The Byte_Source.Seek additionally rejects
      --  an offset past the length.
      Abs_Off : constant U64 := Checked_Add (File.Data_Start, Info.Offset);
      End_Off : constant U64 := Checked_Add (Abs_Off, U64 (Buf_Size));
   begin
      if File.File_Size /= 0 and then End_Off > File.File_Size then
         raise Malformed_GGUF with "Read_Tensor_Raw past end of file";
      end if;
      File.Source.Seek (Interfaces.Unsigned_64 (Abs_Off));
      Read_Exact (File.Source, Buffer, Buf_Size);
   end Read_Tensor_Raw;

   procedure Read_Tensor_Range
     (File        : in out GGUF_File;
      Info        : Tensor_Info;
      Byte_Offset : U64;
      Buffer      : System.Address;
      Buf_Size    : Natural)
   is
      --  Checked 64-bit arithmetic + source-length bound (see Read_Tensor_Raw):
      --  an over-range Byte_Offset must read past the end loud, not silently.
      Abs_Off : constant U64 :=
        Checked_Add (Checked_Add (File.Data_Start, Info.Offset), Byte_Offset);
      End_Off : constant U64 := Checked_Add (Abs_Off, U64 (Buf_Size));
   begin
      if File.File_Size /= 0 and then End_Off > File.File_Size then
         raise Malformed_GGUF with "Read_Tensor_Range past end of file";
      end if;
      File.Source.Seek (Interfaces.Unsigned_64 (Abs_Off));
      Read_Exact (File.Source, Buffer, Buf_Size);
   end Read_Tensor_Range;

   --  Validated byte size, computed once in Open and stored in Info.Byte_Size.
   --  QMatVec synthesises a 1-row sub-tensor on the fly (Byte_Size not set), so
   --  fall back to the checked computation when the stored value is zero.
   function Tensor_Byte_Size (Info : Tensor_Info) return U64 is
   begin
      if Info.Byte_Size /= 0 then
         return Info.Byte_Size;
      end if;
      return Checked_Byte_Size (Info);
   end Tensor_Byte_Size;

   function Tensor_Num_Elements (Info : Tensor_Info) return U64 is
   begin
      return Checked_Num_Elements (Info);
   end Tensor_Num_Elements;

   function Alignment (File : GGUF_File) return U64 is
   begin
      return File.Alignment_Val;
   end Alignment;

   procedure Close (File : in out GGUF_File) is
   begin
      --  Free_Source dispatches Close (releases the fd / connection) and then
      --  deallocates the access; idempotent on null. After this File.Source is
      --  null and the underlying resource is released exactly once.
      LLM_Byte_Source.Free_Source (File.Source);
      File.Is_Open := False;
      File.Tensors.Clear;
      File.Meta.Clear;
   end Close;

   function Is_GGUF (Path : String) return Boolean is
      S     : Source_Access;
      Magic : String (1 .. 4);
      Ok    : Boolean := False;
   begin
      --  Open the source; null means missing / unreadable (not a GGUF).
      S := LLM_Byte_Source.Open_Source (Path);
      if S = null then
         return False;
      end if;
      declare
      begin
         S.Read_Seq (Magic'Address, 4);
         Ok := Magic = "GGUF";
      exception
         when others =>
            Ok := False;   --  short read (< 4 bytes): not a GGUF
      end;
      LLM_Byte_Source.Free_Source (S);
      return Ok;
   end Is_GGUF;

end LLM_GGUF;