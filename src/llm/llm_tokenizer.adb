---------------------------------------------------------------------
-- LLM_Tokenizer body — GPT-2 byte-level BPE (greedy merge by rank)
--
-- Byte-level mode (GGUF tokenizer.ggml.model = "gpt2") maps each input
-- byte through the GPT-2 byte->unicode bijection (e.g. space -> U+0120
-- "Ġ") before BPE, and inverts it on decode, so raw bytes match the
-- UTF-8 vocab pieces. Without a vocab the tokenizer is a 1-id-per-byte
-- fallback.
---------------------------------------------------------------------

with Ada.Containers.Indefinite_Ordered_Maps;
with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;
with Ada.Real_Time;
with Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Unchecked_Deallocation;

package body LLM_Tokenizer is

   NUL : constant Character := Character'Val (0);

   package Str_Int_Maps is new Ada.Containers.Indefinite_Hashed_Maps
     (Key_Type        => String,
      Element_Type    => Integer,
      Hash            => Ada.Strings.Hash,
      Equivalent_Keys => "=");
   Tok_Wall_On : constant Boolean :=
     Ada.Environment_Variables.Exists ("ASPIDA_TOK_WALL");

   package Int_Str_Maps is new Ada.Containers.Indefinite_Ordered_Maps
     (Key_Type => Integer, Element_Type => String);

   type Tokenizer_Data is record
      Vocab      : Str_Int_Maps.Map;   -- piece -> id
      Id2Tok     : Int_Str_Maps.Map;   -- id -> piece
      Merges     : Str_Int_Maps.Map;   -- "left<NUL>right" -> rank
      Loaded     : Boolean := False;
      Byte_Level : Boolean := False;   -- GPT-2 byte->unicode remap
      Gemma_Mode : Boolean := False;   -- SentencePiece: U+2581 space, <0xXX> bytes
      Unk_Id     : Integer := -1;      -- tokenizer.ggml.unknown_token_id (-1 = none)
   end record;

   --  U+2581 "lower one eighth block" — SentencePiece's visible space.
   SP_Space : constant String :=
     Character'Val (16#E2#) & Character'Val (16#96#) & Character'Val (16#81#);

   --------------------------------------------------------------------
   -- GPT-2 byte <-> unicode bijection (computed once at elaboration)
   --------------------------------------------------------------------

   Byte_CP : array (0 .. 255) of Natural;          -- byte -> code point
   CP_Byte : array (0 .. 511) of Integer := [others => -1];  -- code point -> byte

   function Is_Printable (B : Natural) return Boolean is
   begin
      return B in 33 .. 126 or else B in 161 .. 172 or else B in 174 .. 255;
   end Is_Printable;

   --  UTF-8 encode a code point (<= 0x7FF here, so 1 or 2 bytes).
   function CP_UTF8 (CP : Natural) return String is
   begin
      if CP < 16#80# then
         return R : String (1 .. 1) do
            R (1) := Character'Val (CP);
         end return;
      else
         return R : String (1 .. 2) do
            R (1) := Character'Val (16#C0# + CP / 16#40#);
            R (2) := Character'Val (16#80# + CP mod 16#40#);
         end return;
      end if;
   end CP_UTF8;

   --  Map one input byte to its GPT-2 unicode UTF-8 piece.
   function Byte_To_Piece (B : Natural) return String is
   begin
      return CP_UTF8 (Byte_CP (B));
   end Byte_To_Piece;

   function Byte_Level_Piece (Raw : String) return String is
      R : Unbounded_String;
   begin
      for C of Raw loop
         Append (R, Byte_To_Piece (Character'Pos (C)));
      end loop;
      return To_String (R);
   end Byte_Level_Piece;

   --  Invert: a UTF-8 string of GPT-2 code points -> original bytes.
   function Unmap_Bytes (S : String) return String is
      R : Unbounded_String;
      I : Integer := S'First;
   begin
      while I <= S'Last loop
         declare
            Lead : constant Natural := Character'Pos (S (I));
            CP   : Natural;
         begin
            if Lead < 16#80# then
               CP := Lead;
               I := I + 1;
            elsif I < S'Last then
               CP := (Lead mod 16#20#) * 16#40#
                     + (Character'Pos (S (I + 1)) mod 16#40#);
               I := I + 2;
            else
               CP := Lead;        -- malformed tail; pass through
               I := I + 1;
            end if;
            if CP <= CP_Byte'Last and then CP_Byte (CP) >= 0 then
               Append (R, Character'Val (CP_Byte (CP)));
            end if;
         end;
      end loop;
      return To_String (R);
   end Unmap_Bytes;

   --------------------------------------------------------------------
   -- Construction
   --------------------------------------------------------------------

   function Create return Tokenizer is
   begin
      return new Tokenizer_Data;
   end Create;

   procedure Add_Token (T : in out Tokenizer; Piece : String; Id : Integer) is
   begin
      if T = null then
         T := new Tokenizer_Data;
      end if;
      if not T.Vocab.Contains (Piece) then
         T.Vocab.Insert (Piece, Id);
      end if;
      if not T.Id2Tok.Contains (Id) then
         T.Id2Tok.Insert (Id, Piece);
      end if;
   end Add_Token;

   procedure Add_Merge (T : in out Tokenizer; Pair : String; Rank : Integer) is
      Sp : constant Natural := Ada.Strings.Fixed.Index (Pair, " ");
   begin
      if T = null then
         T := new Tokenizer_Data;
      end if;
      if Sp > Pair'First and then Sp < Pair'Last then
         declare
            Left  : constant String := Pair (Pair'First .. Sp - 1);
            Right : constant String := Pair (Sp + 1 .. Pair'Last);
            Key   : constant String := Left & NUL & Right;
         begin
            if not T.Merges.Contains (Key) then
               T.Merges.Insert (Key, Rank);
            end if;
         end;
      end if;
   end Add_Merge;

   procedure Mark_Loaded (T : in out Tokenizer) is
   begin
      if T = null then
         T := new Tokenizer_Data;
      end if;
      T.Loaded := True;
   end Mark_Loaded;

   procedure Load_From_GGUF (T : in out Tokenizer; G : LLM_GGUF.GGUF_File) is
      NT : constant Natural := LLM_GGUF.Token_Count (G);
      NM : constant Natural := LLM_GGUF.Merge_Count (G);
   begin
      if T = null then
         T := new Tokenizer_Data;
      end if;
      for I in 1 .. NT loop
         Add_Token (T, LLM_GGUF.Token_At (G, I), I - 1);  -- ids are 0-based
      end loop;
      for I in 1 .. NM loop
         Add_Merge (T, LLM_GGUF.Merge_At (G, I), I);      -- rank = file order
      end loop;
      declare
         Model : constant String := LLM_GGUF.Metadata (G, "tokenizer.ggml.model");
      begin
         T.Byte_Level := Model = "gpt2";
         T.Gemma_Mode := Model = "gemma4" or else Model = "gemma"
                         or else Model = "llama";
      end;
      --  The model's unknown-token id, used as the fallback for a piece that
      --  is absent from the vocab (instead of the magic 0, a valid token).
      --  Absent for an exhaustive byte-level vocab -> stays -1.
      declare
         U : constant String :=
           LLM_GGUF.Metadata (G, "tokenizer.ggml.unknown_token_id");
      begin
         if U /= "" then
            T.Unk_Id := Integer'Value (U);
         end if;
      exception
         when others => T.Unk_Id := -1;
      end;
      T.Loaded := NT > 0;
   end Load_From_GGUF;

   --------------------------------------------------------------------
   -- Queries
   --------------------------------------------------------------------

   function Is_Loaded (T : Tokenizer) return Boolean is
   begin
      return T /= null and then T.Loaded;
   end Is_Loaded;

   function Vocab_Size (T : Tokenizer) return Integer is
   begin
      if T = null then
         return 0;
      end if;
      return Integer (T.Vocab.Length);
   end Vocab_Size;

   function Unk_Id (T : Tokenizer) return Integer is
     (if T = null then -1 else T.Unk_Id);

   --------------------------------------------------------------------
   -- Decode
   --------------------------------------------------------------------

   --  Gemma/SentencePiece piece -> text: U+2581 becomes a space and a
   --  <0xHH> byte-fallback token becomes its raw byte.
   function Gemma_Unmap (Piece : String) return String is
      R : Unbounded_String;
      I : Integer := Piece'First;
      function Hex_Val (C : Character) return Integer is
        (case C is
            when '0' .. '9' => Character'Pos (C) - Character'Pos ('0'),
            when 'A' .. 'F' => Character'Pos (C) - Character'Pos ('A') + 10,
            when 'a' .. 'f' => Character'Pos (C) - Character'Pos ('a') + 10,
            when others => 0);
   begin
      while I <= Piece'Last loop
         if I + 5 <= Piece'Last
           and then Piece (I .. I + 2) = "<0x" and then Piece (I + 5) = '>'
         then
            Append (R, Character'Val
              (Hex_Val (Piece (I + 3)) * 16 + Hex_Val (Piece (I + 4))));
            I := I + 6;
         elsif I + 2 <= Piece'Last and then Piece (I .. I + 2) = SP_Space then
            Append (R, ' '); I := I + 3;
         else
            Append (R, Piece (I)); I := I + 1;
         end if;
      end loop;
      return To_String (R);
   end Gemma_Unmap;

   function Decode_One (T : Tokenizer; Id : Integer) return String is
   begin
      if T /= null and then T.Id2Tok.Contains (Id) then
         declare
            Piece : constant String := T.Id2Tok.Element (Id);
         begin
            if T.Byte_Level then
               return Unmap_Bytes (Piece);
            elsif T.Gemma_Mode then
               return Gemma_Unmap (Piece);
            else
               return Piece;
            end if;
         end;
      elsif Id in 0 .. 255 then
         return R : String (1 .. 1) do
            R (1) := Character'Val (Id);
         end return;
      else
         return "";
      end if;
   end Decode_One;

   function Token_To_Id (T : Tokenizer; Piece : String) return Integer is
   begin
      if T /= null and then T.Vocab.Contains (Piece) then
         return T.Vocab.Element (Piece);
      else
         return -1;
      end if;
   end Token_To_Id;

   function Decode (T : Tokenizer; Ids : Token_Array) return String is
      R : Unbounded_String;
   begin
      for I in Ids'Range loop
         Append (R, Decode_One (T, Ids (I)));
      end loop;
      return To_String (R);
   end Decode;

   --------------------------------------------------------------------
   -- Encode
   --------------------------------------------------------------------

   function Encode (T : Tokenizer; Text : String) return Token_Array is
      use type Ada.Real_Time.Time;
      TW0 : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;

      function Finish (R : Token_Array) return Token_Array is
      begin
         if Tok_Wall_On and then Text'Length > 4000 then
            Ada.Text_IO.Put_Line
              (Ada.Text_IO.Standard_Error,
               "[TOKWALL] len=" & Integer'Image (Text'Length)
               & " toks=" & Integer'Image (R'Length)
               & " ms=" & Duration'Image
                   (Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - TW0) * 1000.0));
         end if;
         return R;
      end Finish;
   begin
      --  Cap input length first: even at O(N log N) a pathological input
      --  should be rejected loudly before the engine's context clamp runs
      --  (the cap is far above any real context window).
      if Text'Length > Max_Encode_Len then
         raise Input_Too_Long
           with "Encode input length" & Integer'Image (Text'Length)
                & " exceeds cap" & Integer'Image (Max_Encode_Len);
      end if;

      -- Byte-level fallback when no vocabulary is loaded.
      if not Is_Loaded (T) then
         return R : Token_Array (1 .. Text'Length) do
            for I in 1 .. Text'Length loop
               R (I) := Character'Pos (Text (Text'First + I - 1));
            end loop;
         end return;
      end if;

      if Text'Length = 0 then
         return Empty : Token_Array (1 .. 0);
      end if;

      -- Greedy byte-pair encoding by merge rank. The initial pieces are
      -- appended straight into the working text WU with their lengths in a
      -- heap array (one input character yields at most one piece — a multi-
      -- byte UTF-8 char that falls back to byte tokens yields one piece per
      -- BYTE, still bounded by Text'Length): no per-piece Unbounded_String
      -- array on the task stack (a 100 KB prompt would overflow it).
      declare
         type Nat_Arr is array (Positive range <>) of Natural;
         type Nat_Ptr is access Nat_Arr;
         procedure Free is new Ada.Unchecked_Deallocation (Nat_Arr, Nat_Ptr);

         Lens : Nat_Ptr := new Nat_Arr (1 .. Integer'Max (1, Text'Length));
         WU   : Unbounded_String := Null_Unbounded_String;
         N    : Natural := 0;

         procedure Add_Piece (P : String) is
         begin
            N := N + 1;
            Lens (N) := P'Length;
            Append (WU, P);
         end Add_Piece;

         function Byte_Token (B : Natural) return String is
            Hx : constant String := "0123456789ABCDEF";
         begin
            return "<0x" & Hx (B / 16 + 1) & Hx (B mod 16 + 1) & ">";
         end Byte_Token;
      begin
         if T.Gemma_Mode then
            --  SentencePiece: space -> U+2581, then per-UTF-8-character initial
            --  pieces, falling back to <0xHH> byte tokens for unknown chars.
            declare
               I : Integer := Text'First;
            begin
               while I <= Text'Last loop
                  if Text (I) = ' ' then
                     Add_Piece (SP_Space);
                     I := I + 1;
                  else
                     declare
                        Lead : constant Natural := Character'Pos (Text (I));
                        CL   : constant Integer :=
                          (if    Lead < 16#80#  then 1
                           elsif Lead >= 16#F0# then 4
                           elsif Lead >= 16#E0# then 3
                           elsif Lead >= 16#C0# then 2 else 1);
                        LB   : constant Integer := Integer'Min (Text'Last, I + CL - 1);
                        Ch   : constant String := Text (I .. LB);
                     begin
                        if T.Vocab.Contains (Ch) then
                           Add_Piece (Ch);
                        else
                           for J in I .. LB loop
                              Add_Piece (Byte_Token (Character'Pos (Text (J))));
                           end loop;
                        end if;
                        I := LB + 1;
                     end;
                  end if;
               end loop;
            end;
         else
            for I in 1 .. Text'Length loop
               declare
                  B : constant Natural := Character'Pos (Text (Text'First + I - 1));
               begin
                  if T.Byte_Level then
                     Add_Piece (Byte_To_Piece (B));
                  else
                     Add_Piece (Text (Text'First + I - 1 .. Text'First + I - 1));
                  end if;
               end;
            end loop;
         end if;

         --  Greedy merge by rank, O(N log N): a doubly-linked list of symbols
         --  (each a slice of the concatenated working text W — merges of
         --  adjacent slices are just span extensions, no string building) and
         --  a min-heap of candidate pairs ordered by (rank, position). Stale
         --  heap entries (a side already merged away) are detected by
         --  comparing the recorded span snapshots and skipped. This replaces
         --  a scan-all-pairs-per-merge loop that was O(N^2) map lookups with
         --  a fresh key string each — ~90+ seconds for a 20 KB prompt, which
         --  made every real agent request (25-100 KB of context) time out.
         --  The merge ORDER is identical: global lowest rank first, leftmost
         --  on ties — so the output tokens are bit-identical to the old loop.
         declare
         begin
            declare
               W : constant String := To_String (WU);

               type Sym is record
                  From, To : Integer := 0;   -- span in W
                  Prev, Nxt : Natural := 0;  -- linked list, 0 = none
                  Alive     : Boolean := False;
               end record;
               type Sym_Arr is array (Natural range <>) of Sym;
               type Sym_Ptr is access Sym_Arr;
               procedure Free is new Ada.Unchecked_Deallocation (Sym_Arr, Sym_Ptr);

               type Cand is record
                  Rank   : Integer := 0;
                  L      : Natural := 0;     -- left symbol index
                  LT, RT : Integer := 0;     -- span snapshots for staleness
               end record;
               type Cand_Arr is array (Positive range <>) of Cand;
               type Cand_Ptr is access Cand_Arr;
               procedure Free is new Ada.Unchecked_Deallocation (Cand_Arr, Cand_Ptr);

               S  : Sym_Ptr := new Sym_Arr (1 .. N);
               --  Each merge enqueues at most 2 new candidates; N-1 initial.
               H  : Cand_Ptr := new Cand_Arr (1 .. 3 * N + 8);
               HN : Natural := 0;

               function Less (A, B : Cand) return Boolean is
                 (A.Rank < B.Rank
                  or else (A.Rank = B.Rank and then A.L < B.L));

               procedure Push (C : Cand) is
                  I : Natural := HN + 1;
               begin
                  HN := I;
                  H (I) := C;
                  while I > 1 and then Less (H (I), H (I / 2)) loop
                     declare
                        Tmp : constant Cand := H (I / 2);
                     begin
                        H (I / 2) := H (I); H (I) := Tmp;
                     end;
                     I := I / 2;
                  end loop;
               end Push;

               procedure Pop (C : out Cand) is
                  I : Natural := 1;
               begin
                  C := H (1);
                  H (1) := H (HN);
                  HN := HN - 1;
                  loop
                     declare
                        Sm : Natural := I;
                     begin
                        if 2 * I <= HN and then Less (H (2 * I), H (Sm)) then
                           Sm := 2 * I;
                        end if;
                        if 2 * I + 1 <= HN and then Less (H (2 * I + 1), H (Sm))
                        then
                           Sm := 2 * I + 1;
                        end if;
                        exit when Sm = I;
                        declare
                           Tmp : constant Cand := H (Sm);
                        begin
                           H (Sm) := H (I); H (I) := Tmp;
                        end;
                        I := Sm;
                     end;
                  end loop;
               end Pop;

               --  Queue (L, Nxt(L)) if that pair has a merge rank.
               procedure Try_Push (L : Natural) is
               begin
                  if L = 0 or else S (L).Nxt = 0 then
                     return;
                  end if;
                  declare
                     R   : constant Natural := S (L).Nxt;
                     Key : constant String :=
                       W (S (L).From .. S (L).To) & NUL
                       & W (S (R).From .. S (R).To);
                  begin
                     declare
                        Cu : constant Str_Int_Maps.Cursor := T.Merges.Find (Key);
                     begin
                        if Str_Int_Maps.Has_Element (Cu) then
                           Push ((Rank => Str_Int_Maps.Element (Cu), L => L,
                                  LT => S (L).To, RT => S (R).To));
                        end if;
                     end;
                  end;
               end Try_Push;

               Pos : Integer := W'First;
            begin
               for I in 1 .. N loop
                  S (I) := (From => Pos, To => Pos + Lens (I) - 1,
                            Prev => (if I > 1 then I - 1 else 0),
                            Nxt  => (if I < N then I + 1 else 0),
                            Alive => True);
                  Pos := Pos + Lens (I);
               end loop;
               for I in 1 .. N - 1 loop
                  Try_Push (I);
               end loop;

               while HN > 0 loop
                  declare
                     C : Cand;
                  begin
                     Pop (C);
                     declare
                        L : constant Natural := C.L;
                        R : constant Natural :=
                          (if S (L).Alive then S (L).Nxt else 0);
                     begin
                        --  Merges always deactivate the RIGHT symbol, so the
                        --  span snapshots fully determine staleness.
                        if R /= 0 and then S (R).Alive
                          and then S (L).To = C.LT and then S (R).To = C.RT
                        then
                           S (L).To    := S (R).To;
                           S (R).Alive := False;
                           S (L).Nxt   := S (R).Nxt;
                           if S (R).Nxt /= 0 then
                              S (S (R).Nxt).Prev := L;
                           end if;
                           Try_Push (S (L).Prev);
                           Try_Push (L);
                        end if;
                     end;
                  end;
               end loop;

               --  Walk the surviving symbols (symbol 1 is never the right
               --  side of a merge, so it is always the list head).
               declare
                  Cnt : Natural := 0;
                  I   : Natural := 1;
               begin
                  while I /= 0 loop
                     Cnt := Cnt + 1;
                     I := S (I).Nxt;
                  end loop;
                  declare
                     R2 : Token_Array (1 .. Cnt);
                     K  : Natural := 0;
                  begin
                     I := 1;
                     while I /= 0 loop
                        K := K + 1;
                        declare
                           P : constant String := W (S (I).From .. S (I).To);
                        begin
                           declare
                              Cu : constant Str_Int_Maps.Cursor :=
                                T.Vocab.Find (P);
                           begin
                              if Str_Int_Maps.Has_Element (Cu) then
                                 R2 (K) := Str_Int_Maps.Element (Cu);
                                 goto Looked_Up;
                              end if;
                           end;
                           if T.Unk_Id >= 0 then
                              R2 (K) := T.Unk_Id;  -- the model's real UNK token
                           else
                              R2 (K) := 0;         -- last resort (no UNK)
                           end if;
                           <<Looked_Up>>
                        end;
                        I := S (I).Nxt;
                     end loop;
                     Free (S);
                     Free (H);
                     Free (Lens);
                     return Finish (R2);
                  end;
               end;
            exception
               when others =>
                  Free (S);
                  Free (H);
                  Free (Lens);
                  raise;
            end;
         end;
      end;
   end Encode;

begin
   --  Build the GPT-2 byte<->unicode bijection once.
   declare
      N : Natural := 0;
   begin
      for B in 0 .. 255 loop
         if Is_Printable (B) then
            Byte_CP (B) := B;
         else
            Byte_CP (B) := 256 + N;
            N := N + 1;
         end if;
      end loop;
      for B in 0 .. 255 loop
         CP_Byte (Byte_CP (B)) := B;
      end loop;
   end;
end LLM_Tokenizer;
