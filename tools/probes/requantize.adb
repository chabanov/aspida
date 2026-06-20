------------------------------------------------------------------------
-- requantize — convert any GGUF the engine can read into a smaller/different
-- weight-quant format the engine can serve. Byte-level rewriter: the metadata
-- section is copied VERBATIM (so every KV — arch, hyperparams, tokenizer, chat
-- template, rope config — survives type-exact), only the tensor-info table
-- (types + offsets) and tensor data are rewritten. 2D, block-aligned weight
-- tensors are dequantized to F32 then re-quantized to the target; norms / 1D /
-- non-dequantizable tensors are copied through unchanged.
--
--   requantize <src.gguf> <dst.gguf> <q4_k|q5_k|q6_k|q8_0|q4_0|q5_0|q4_k_m|q5_k_m>
--   (the _k_m variants keep the base format but route output/attn_v/ffn_down
--    tensors to Q6_K, matching llama.cpp's higher-quality "medium" mixes)
------------------------------------------------------------------------

with Ada.Command_Line;       use Ada.Command_Line;
with Ada.Text_IO;            use Ada.Text_IO;
with Ada.Streams.Stream_IO;
with Ada.Streams;            use Ada.Streams;
with Ada.Strings.Unbounded;  use Ada.Strings.Unbounded;
with Ada.Characters.Handling; use Ada.Characters.Handling;
with Ada.Unchecked_Deallocation;
with Interfaces;             use Interfaces;
with LLM_GGUF;               use LLM_GGUF;
with LLM_Dequant;
with LLM_Quant;
with LLM_Tensor;             use LLM_Tensor;

procedure Requantize is

   package SIO renames Ada.Streams.Stream_IO;

   type Str_Ptr is access String;
   procedure Free is new Ada.Unchecked_Deallocation (String, Str_Ptr);

   --  little-endian byte emitters (1 char == 1 byte)
   function B_U32 (V : Unsigned_32) return String is
      S : String (1 .. 4); X : Unsigned_32 := V;
   begin
      for I in 1 .. 4 loop
         S (I) := Character'Val (Natural (X and 16#FF#)); X := Shift_Right (X, 8);
      end loop;
      return S;
   end B_U32;

   function B_U64 (V : Unsigned_64) return String is
      S : String (1 .. 8); X : Unsigned_64 := V;
   begin
      for I in 1 .. 8 loop
         S (I) := Character'Val (Natural (X and 16#FF#)); X := Shift_Right (X, 8);
      end loop;
      return S;
   end B_U64;

   function B_Str (S : String) return String is (B_U64 (Unsigned_64 (S'Length)) & S);

   --  GGML enum -> wire code (the enum positions are NOT the codes).
   function Code_Of (K : GGML_Type) return Unsigned_32 is
   begin
      case K is
         when GGML_TYPE_F32  => return 0;   when GGML_TYPE_F16  => return 1;
         when GGML_TYPE_Q4_0 => return 2;   when GGML_TYPE_Q4_1 => return 3;
         when GGML_TYPE_Q5_0 => return 6;   when GGML_TYPE_Q5_1 => return 7;
         when GGML_TYPE_Q8_0 => return 8;   when GGML_TYPE_Q8_1 => return 9;
         when GGML_TYPE_Q2_K => return 10;  when GGML_TYPE_Q3_K => return 11;
         when GGML_TYPE_Q4_K => return 12;  when GGML_TYPE_Q5_K => return 13;
         when GGML_TYPE_Q6_K => return 14;  when GGML_TYPE_Q8_K => return 15;
         when GGML_TYPE_BF16 => return 30;
         when GGML_TYPE_UNKNOWN =>
            --  Read-side sentinel for an unsupported type; a writer never
            --  serializes it.
            raise Program_Error
              with "cannot requantize an unknown/unsupported ggml type";
      end case;
   end Code_Of;

   --  Target format -> (base wire code, super-block element count, mix flag).
   --  The "_k_m" variants keep the base for most tensors but bump the
   --  quality-sensitive ones (output / attn_v / ffn_down) to Q6_K, like
   --  llama.cpp's Q4_K_M / Q5_K_M recipes — big quality gain, small size cost.
   procedure Parse_Target (S : String; Code : out Unsigned_32; Blk : out Natural;
                           Mix : out Boolean; Ok : out Boolean) is
      T : constant String := To_Lower (S);
   begin
      Ok := True; Mix := False;
      if    T = "q4_k"   then Code := 12; Blk := 256;
      elsif T = "q5_k"   then Code := 13; Blk := 256;
      elsif T = "q6_k"   then Code := 14; Blk := 256;
      elsif T = "q8_0"   then Code := 8;  Blk := 32;
      elsif T = "q4_0"   then Code := 2;  Blk := 32;
      elsif T = "q5_0"   then Code := 6;  Blk := 32;
      elsif T = "q4_k_m" then Code := 12; Blk := 256; Mix := True;
      elsif T = "q5_k_m" then Code := 13; Blk := 256; Mix := True;
      else  Code := 0; Blk := 0; Ok := False;
      end if;
   end Parse_Target;

   function Kind_Of_Code (Code : Unsigned_32) return GGML_Type is
     (case Code is
         when 12 => GGML_TYPE_Q4_K, when 13 => GGML_TYPE_Q5_K,
         when 14 => GGML_TYPE_Q6_K, when 8 => GGML_TYPE_Q8_0,
         when 2  => GGML_TYPE_Q4_0, when others => GGML_TYPE_Q5_0);

   --  A _K_M mix routes these (high-error-sensitivity) tensors to Q6_K, like
   --  llama.cpp: the LM head (exactly "output.weight" — NOT blk.*.attn_output)
   --  plus every attn_v and ffn_down projection.
   function Sensitive (Name : String) return Boolean is
      function Tail (Suf : String) return Boolean is
        (Name'Length >= Suf'Length
         and then Name (Name'Last - Suf'Length + 1 .. Name'Last) = Suf);
   begin
      return Name = "output.weight"
        or else Tail ("attn_v.weight")
        or else Tail ("ffn_down.weight");
   end Sensitive;

   function Requant (T : Tensor; Code : Unsigned_32) return String is
   begin
      case Code is
         when 12 => return LLM_Quant.Quantize_Q4_K (T);
         when 13 => return LLM_Quant.Quantize_Q5_K (T);
         when 14 => return LLM_Quant.Quantize_Q6_K (T);
         when 8  => return LLM_Quant.Quantize_Q8_0 (T);
         when 2  => return LLM_Quant.Quantize_Q4_0 (T);
         when others => return LLM_Quant.Quantize_Q5_0 (T);  -- 6
      end case;
   end Requant;

   Align : Natural := 32;
   function Round_Up (N : Natural) return Natural is
     (((N + Align - 1) / Align) * Align);

   --  Read the source file's metadata section (header skipped) VERBATIM by
   --  walking the KVs, so types/values are preserved byte-exact.
   function Read_Meta_Blob (Path : String; N_Meta : Natural) return String is
      F    : SIO.File_Type;
      Blob : Unbounded_String;

      --  Read N bytes from F, append them to Blob (verbatim copy).
      procedure Eat (N : Natural) is
         Buf  : Stream_Element_Array (1 .. Stream_Element_Offset (Integer'Max (N, 1)));
         Last : Stream_Element_Offset;
      begin
         if N = 0 then return; end if;
         SIO.Read (F, Buf (1 .. Stream_Element_Offset (N)), Last);
         declare R : String (1 .. Natural (Last)); begin
            for I in R'Range loop
               R (I) := Character'Val (Natural (Buf (Stream_Element_Offset (I))));
            end loop;
            Append (Blob, R);
         end;
      end Eat;

      --  Same, but also decode the little-endian integer just read.
      function RU32 return Unsigned_32 is
         Before : constant Natural := Length (Blob);
      begin
         Eat (4);
         return Unsigned_32 (Character'Pos (Element (Blob, Before + 1)))
           or Shift_Left (Unsigned_32 (Character'Pos (Element (Blob, Before + 2))), 8)
           or Shift_Left (Unsigned_32 (Character'Pos (Element (Blob, Before + 3))), 16)
           or Shift_Left (Unsigned_32 (Character'Pos (Element (Blob, Before + 4))), 24);
      end RU32;

      function RU64 return Unsigned_64 is
         Before : constant Natural := Length (Blob); V : Unsigned_64 := 0;
      begin
         Eat (8);
         for I in reverse 1 .. 8 loop
            V := Shift_Left (V, 8)
               or Unsigned_64 (Character'Pos (Element (Blob, Before + I)));
         end loop;
         return V;
      end RU64;

      procedure Read_Value (T : Unsigned_32) is
      begin
         case T is
            when 0 | 1 | 7    => Eat (1);
            when 2 | 3        => Eat (2);
            when 4 | 5 | 6    => Eat (4);
            when 10 | 11 | 12 => Eat (8);
            when 8            => Eat (Natural (RU64));           -- string
            when 9 =>                                            -- array
               declare ET  : constant Unsigned_32 := RU32;
                       Cnt : constant Unsigned_64 := RU64;
               begin
                  for I in 1 .. Natural (Cnt) loop Read_Value (ET); end loop;
               end;
            when others =>
               raise Constraint_Error with "unknown GGUF meta type" & T'Image;
         end case;
      end Read_Value;

   begin
      SIO.Open (F, SIO.In_File, Path);
      Eat (24);                              -- header
      Blob := Null_Unbounded_String;         -- ...discard it; blob = KV section only
      for K in 1 .. N_Meta loop
         Eat (Natural (RU64));               -- key string (u64 len + bytes)
         Read_Value (RU32);                  -- value type + value
      end loop;
      SIO.Close (F);
      return To_String (Blob);
   end Read_Meta_Blob;

begin
   if Argument_Count < 3 then
      Put_Line ("usage: requantize <src.gguf> <dst.gguf> "
                & "<q4_k|q5_k|q6_k|q8_0|q4_0|q5_0|q4_k_m|q5_k_m>");
      return;
   end if;

   declare
      Src    : constant String := Argument (1);
      Dst    : constant String := Argument (2);
      T_Code : Unsigned_32; T_Blk : Natural; Ok : Boolean; Mix : Boolean;
   begin
      Parse_Target (Argument (3), T_Code, T_Blk, Mix, Ok);
      if not Ok then Put_Line ("bad target format: " & Argument (3)); return; end if;

      declare
         G : GGUF_File;
      begin
         Open (G, Src);
         if not Is_Open (G) then Put_Line ("cannot open " & Src); return; end if;
         Align := Natural (Alignment (G));

         declare
            NT   : constant Natural := Tensor_Count (G);
            NM   : constant Natural := Metadata_Count (G);
            Blob : constant String  := Read_Meta_Blob (Src, NM);

            New_Code : array (1 .. NT) of Unsigned_32;
            New_Size : array (1 .. NT) of Natural;
            New_Off  : array (1 .. NT) of Natural;
            Do_Req   : array (1 .. NT) of Boolean;
            Off      : Natural := 0;
            N_Req    : Natural := 0;
         begin
            --  Pass 1: decide per-tensor target + lay out aligned offsets.
            for I in 1 .. NT loop
               declare
                  Info  : constant Tensor_Info := Tensor_At (G, I);
                  --  Per-tensor target: a _K_M mix bumps sensitive tensors to Q6_K.
                  TCode : constant Unsigned_32 :=
                    (if Mix and then Sensitive (To_String (Info.Name)) then 14
                     else T_Code);
                  Elig  : constant Boolean :=
                    Info.N_Dims = 2
                    and then Natural (Info.Dims (1)) mod T_Blk = 0
                    and then LLM_Dequant.Is_Supported (Info.Kind)
                    and then Code_Of (Info.Kind) /= TCode;
                  Tgt   : Tensor_Info := Info;
               begin
                  Do_Req (I) := Elig;
                  if Elig then
                     N_Req := N_Req + 1;
                     New_Code (I) := TCode;
                     Tgt.Kind     := Kind_Of_Code (TCode);  -- size at target type
                     New_Size (I) := Natural (Tensor_Byte_Size (Tgt));
                  else
                     New_Code (I) := Code_Of (Info.Kind);
                     New_Size (I) := Natural (Tensor_Byte_Size (Info));
                  end if;
                  New_Off (I) := Off;
                  Off := Round_Up (Off + New_Size (I));
               end;
            end loop;

            Put_Line ("requantize: " & NT'Image & " tensors,"
                      & N_Req'Image & " -> " & To_Lower (Argument (3))
                      & " (" & Natural'Image (NT - N_Req) & " copied as-is)");

            --  Build header + tensor-info table.
            declare
               F   : SIO.File_Type;
               TI  : Unbounded_String;

               --  Chunked so a 100MB+ tensor never lands a giant array on the
               --  stack (the String arg is by-reference; only Buf is on stack).
               procedure Put_S (S : String) is
                  Chunk : constant := 65536;
                  I     : Natural := S'First;
               begin
                  while I <= S'Last loop
                     declare
                        Hi  : constant Natural := Natural'Min (I + Chunk - 1, S'Last);
                        Buf : Stream_Element_Array
                                (1 .. Stream_Element_Offset (Hi - I + 1));
                     begin
                        for J in I .. Hi loop
                           Buf (Stream_Element_Offset (J - I + 1)) :=
                             Stream_Element (Character'Pos (S (J)));
                        end loop;
                        SIO.Write (F, Buf);
                        I := Hi + 1;
                     end;
                  end loop;
               end Put_S;

               procedure Pad (N : Natural) is
               begin if N > 0 then Put_S ([1 .. N => Character'Val (0)]); end if; end Pad;
            begin
               for I in 1 .. NT loop
                  declare Info : constant Tensor_Info := Tensor_At (G, I); begin
                     Append (TI, B_Str (To_String (Info.Name)));
                     Append (TI, B_U32 (Unsigned_32 (Info.N_Dims)));
                     for D in 1 .. Natural (Info.N_Dims) loop
                        Append (TI, B_U64 (Unsigned_64 (Info.Dims (D))));
                     end loop;
                     Append (TI, B_U32 (New_Code (I)));
                     Append (TI, B_U64 (Unsigned_64 (New_Off (I))));
                  end;
               end loop;

               SIO.Create (F, SIO.Out_File, Dst);
               Put_S ("GGUF" & B_U32 (3)
                      & B_U64 (Unsigned_64 (NT)) & B_U64 (Unsigned_64 (NM)));
               Put_S (Blob);
               Put_S (To_String (TI));

               declare
                  Pos   : constant Natural := 24 + Blob'Length + Length (TI);
                  DStart : constant Natural := Round_Up (Pos);
                  Cursor : Natural := 0;
               begin
                  Pad (DStart - Pos);
                  for I in 1 .. NT loop
                     declare
                        Info : constant Tensor_Info := Tensor_At (G, I);
                        SSz  : constant Natural := Natural (Tensor_Byte_Size (Info));
                        Raw  : Str_Ptr := new String (1 .. SSz);  -- heap (can be 100MB+)
                     begin
                        Read_Tensor_Raw (G, Info, Raw.all'Address, SSz);
                        Pad (New_Off (I) - Cursor);
                        Cursor := New_Off (I);
                        if Do_Req (I) then
                           --  Requantize one ROW at a time: Quantize_* allocates
                           --  its whole result on the stack, so a full 100MB+
                           --  weight would overflow. ne0-sized rows keep it tiny.
                           declare
                              Ne1 : constant Natural := Natural (Info.Dims (2));
                              RB  : constant Natural := SSz / Ne1;   -- src bytes/row
                              RInfo : Tensor_Info := Info;
                           begin
                              RInfo.N_Dims := 2;
                              RInfo.Dims   := [Info.Dims (1), 1, 0, 0];
                              for R in 0 .. Ne1 - 1 loop
                                 declare
                                    Lo : constant Natural := 1 + R * RB;
                                    Tr : constant Tensor := LLM_Dequant.Dequantize
                                      (RInfo, Raw.all (Lo .. Lo + RB - 1));
                                    NB : constant String := Requant (Tr, New_Code (I));
                                 begin
                                    Put_S (NB);
                                    Cursor := Cursor + NB'Length;
                                 end;
                              end loop;
                           end;
                        else
                           Put_S (Raw.all);
                           Cursor := Cursor + SSz;
                        end if;
                        Free (Raw);
                     end;
                  end loop;
               end;
               SIO.Close (F);
            end;
         end;
         Close (G);
      end;
      Put_Line ("wrote " & Dst);
   end;
end Requantize;
