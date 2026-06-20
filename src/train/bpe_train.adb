---------------------------------------------------------------------
-- BPE_Train body.
---------------------------------------------------------------------

with Ada.Strings.Unbounded;             use Ada.Strings.Unbounded;
with Ada.Containers;                    use Ada.Containers;
with Ada.Containers.Vectors;
with Ada.Containers.Hashed_Maps;
with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;
with Interfaces;                        use Interfaces;

package body BPE_Train is

   --  A symbol is a token id; a word is a sequence of symbols with a frequency.
   package Sym_Vectors is new Ada.Containers.Vectors (Positive, Natural);
   use Sym_Vectors;

   type Word is record
      Syms  : Sym_Vectors.Vector;
      Count : Natural := 0;
   end record;
   package Word_Vectors is new Ada.Containers.Vectors (Positive, Word);

   package Piece_Vectors is new
     Ada.Containers.Vectors (Natural, Unbounded_String);

   type Merge_Rec is record
      A, B, New_Id : Natural;
   end record;
   package Merge_Vectors is new Ada.Containers.Vectors (Positive, Merge_Rec);

   --  Pair (A,B) packed into one 64-bit code, for counting and for the encode
   --  lookup of "is there a merge for this pair, and at what rank".
   function Code (A, B : Natural) return Unsigned_64 is
     (Shift_Left (Unsigned_64 (A), 32) or Unsigned_64 (B));

   function H64 (K : Unsigned_64) return Hash_Type is
     (Hash_Type'Mod (K xor Shift_Right (K, 29) xor Shift_Left (K, 17)));

   package Count_Maps is new Ada.Containers.Hashed_Maps
     (Unsigned_64, Natural, H64, "=");

   type Pair_Info is record
      Rank   : Positive;
      New_Id : Natural;
   end record;
   package Pair_Maps is new Ada.Containers.Hashed_Maps
     (Unsigned_64, Pair_Info, H64, "=");

   package Freq_Maps is new Ada.Containers.Indefinite_Hashed_Maps
     (String, Natural, Ada.Strings.Hash, "=");

   type Trainer_Data is record
      Pieces : Piece_Vectors.Vector;   -- id -> literal byte string
      Merges : Merge_Vectors.Vector;
      PMap   : Pair_Maps.Map;          -- pair code -> (rank, new_id)
   end record;

   ------------------------------------------------------------------
   --  Pre-tokenization (GPT-2 / LLaMA byte-level convention): a space starts a
   --  new word AND is kept as that word's leading byte, so "the cat" splits as
   --  ["the", " cat"] — every word after the first carries a leading space
   --  (the raw 0x20 byte; this engine stores pieces as literal bytes, not the
   --  GPT-2 Ġ unicode remap, which the inference tokenizer applies separately).
   --  The first word gets NO leading space: adding one would make Decode emit a
   --  phantom leading space and break the lossless Encode∘Decode = identity
   --  round-trip (the hard constraint). Merges never span a space because each
   --  space begins a fresh word. Encode (below) uses this same split, so Train
   --  and Encode stay consistent.
   ------------------------------------------------------------------
   procedure Pretokenize (Text : String; Freq : out Freq_Maps.Map) is
      Cur : Unbounded_String;
      procedure Flush is
      begin
         if Length (Cur) > 0 then
            declare
               W : constant String := To_String (Cur);
            begin
               if Freq.Contains (W) then
                  Freq.Replace (W, Freq.Element (W) + 1);
               else
                  Freq.Insert (W, 1);
               end if;
            end;
            Cur := Null_Unbounded_String;
         end if;
      end Flush;
   begin
      Freq.Clear;
      for I in Text'Range loop
         if Text (I) = ' ' then
            Flush;        -- a space begins a fresh word
         end if;
         Append (Cur, Text (I));
      end loop;
      Flush;
   end Pretokenize;

   --  Symbols of a word string = its raw bytes (ids 0 .. 255).
   function Bytes_Of (W : String) return Sym_Vectors.Vector is
   begin
      return V : Sym_Vectors.Vector do
         for C of W loop V.Append (Character'Pos (C)); end loop;
      end return;
   end Bytes_Of;

   --  Replace every non-overlapping adjacent (A,B) in S with New_Id.
   procedure Apply_Merge
     (S : in out Sym_Vectors.Vector; A, B, New_Id : Natural)
   is
      Out_V : Sym_Vectors.Vector;
      I     : Natural := S.First_Index;
   begin
      while I <= S.Last_Index loop
         if I < S.Last_Index
           and then S (I) = A and then S (I + 1) = B
         then
            Out_V.Append (New_Id);
            I := I + 2;
         else
            Out_V.Append (S (I));
            I := I + 1;
         end if;
      end loop;
      S := Out_V;
   end Apply_Merge;

   ------------------------------------------------------------------
   --  Train
   ------------------------------------------------------------------
   procedure Train
     (T : out Trainer; Corpus : String; Target_Vocab : Positive)
   is
      Target : constant Natural := Natural'Max (256, Target_Vocab);
      Freq   : Freq_Maps.Map;
      Words  : Word_Vectors.Vector;
   begin
      T := new Trainer_Data;

      --  Base alphabet: one token per byte value.
      for B in 0 .. 255 loop
         T.Pieces.Append (To_Unbounded_String ([1 => Character'Val (B)]));
      end loop;

      --  Build the (unique word -> count) table, then the working word list.
      Pretokenize (Corpus, Freq);
      for C in Freq.Iterate loop
         Words.Append (Word'(Syms  => Bytes_Of (Freq_Maps.Key (C)),
                             Count => Freq_Maps.Element (C)));
      end loop;

      --  Greedily merge the most frequent adjacent pair until Target is hit.
      while Natural (T.Pieces.Length) < Target loop
         declare
            Counts    : Count_Maps.Map;
            Best_Code : Unsigned_64 := 0;
            Best_Cnt  : Natural := 0;
            Best_Pair : Unbounded_String;   -- lexicographic tie-break key
            Found     : Boolean := False;
         begin
            for W of Words loop
               if W.Syms.Last_Index > W.Syms.First_Index then
                  for I in W.Syms.First_Index .. W.Syms.Last_Index - 1 loop
                     declare
                        K : constant Unsigned_64 :=
                          Code (W.Syms (I), W.Syms (I + 1));
                        Prev : constant Natural :=
                          (if Counts.Contains (K) then Counts.Element (K)
                           else 0);
                     begin
                        if Counts.Contains (K) then
                           Counts.Replace (K, Prev + W.Count);
                        else
                           Counts.Insert (K, W.Count);
                        end if;
                     end;
                  end loop;
               end if;
            end loop;

            --  Pick the most frequent pair. On a tie, choose the
            --  lexicographically-smaller pair (by concatenated piece bytes)
            --  rather than whichever the hash-map iterator visits first —
            --  hash iteration order is container/compiler-dependent, so an
            --  unqualified tie-break would make the learned vocab differ across
            --  builds for the same corpus. This makes training reproducible.
            for C in Counts.Iterate loop
               declare
                  Cnt  : constant Natural   := Count_Maps.Element (C);
                  K    : constant Unsigned_64 := Count_Maps.Key (C);
                  A    : constant Natural   := Natural (Shift_Right (K, 32));
                  B    : constant Natural   := Natural (K and 16#FFFF_FFFF#);
                  Cand : constant String    :=
                    To_String (T.Pieces (A)) & To_String (T.Pieces (B));
               begin
                  if Cnt > Best_Cnt
                    or else (Cnt = Best_Cnt and then Found
                             and then Cand < To_String (Best_Pair))
                  then
                     Best_Cnt  := Cnt;
                     Best_Code := K;
                     Best_Pair := To_Unbounded_String (Cand);
                     Found     := True;
                  end if;
               end;
            end loop;

            exit when not Found or else Best_Cnt = 0;

            declare
               A      : constant Natural :=
                 Natural (Shift_Right (Best_Code, 32));
               B      : constant Natural :=
                 Natural (Best_Code and 16#FFFF_FFFF#);
               New_Id : constant Natural := Natural (T.Pieces.Length);  -- next index
            begin
               T.Pieces.Append (T.Pieces.Element (A) & T.Pieces.Element (B));
               T.Merges.Append (Merge_Rec'(A => A, B => B, New_Id => New_Id));
               T.PMap.Insert
                 (Code (A, B),
                  Pair_Info'(Rank => Positive (T.Merges.Length),
                             New_Id => New_Id));
               for W of Words loop
                  Apply_Merge (W.Syms, A, B, New_Id);
               end loop;
            end;
         end;
      end loop;
   end Train;

   function Vocab_Size (T : Trainer) return Natural is
     (Natural (T.Pieces.Length));

   function Num_Merges (T : Trainer) return Natural is
     (Natural (T.Merges.Length));

   function Token_Piece (T : Trainer; Id : Natural) return String is
     (To_String (T.Pieces (Id)));

   function Merge_Left_Id (T : Trainer; Index : Positive) return Natural is
     (T.Merges (Index).A);

   function Merge_Right_Id (T : Trainer; Index : Positive) return Natural is
     (T.Merges (Index).B);

   ------------------------------------------------------------------
   --  Encode / Decode
   ------------------------------------------------------------------
   --  Greedy: repeatedly apply the lowest-rank merge present in the word.
   procedure Encode_Word (T : Trainer; W : String; Out_V : in out Sym_Vectors.Vector)
   is
      S : Sym_Vectors.Vector := Bytes_Of (W);
   begin
      loop
         declare
            Best_Rank : Natural := 0;          -- 0 = "none yet"
            BA, BB, BN : Natural := 0;
            Found : Boolean := False;
         begin
            if S.Last_Index > S.First_Index then
               for I in S.First_Index .. S.Last_Index - 1 loop
                  declare
                     K : constant Unsigned_64 := Code (S (I), S (I + 1));
                  begin
                     if T.PMap.Contains (K) then
                        declare
                           PI : constant Pair_Info := T.PMap.Element (K);
                        begin
                           if not Found or else PI.Rank < Best_Rank then
                              Best_Rank := PI.Rank;
                              BA := S (I); BB := S (I + 1); BN := PI.New_Id;
                              Found := True;
                           end if;
                        end;
                     end if;
                  end;
               end loop;
            end if;
            exit when not Found;
            Apply_Merge (S, BA, BB, BN);
         end;
      end loop;
      for Sym of S loop Out_V.Append (Sym); end loop;
   end Encode_Word;

   function Encode (T : Trainer; Text : String) return Id_Array is
      Out_V : Sym_Vectors.Vector;
      Cur   : Unbounded_String;

      procedure Flush is
      begin
         if Length (Cur) > 0 then
            Encode_Word (T, To_String (Cur), Out_V);
            Cur := Null_Unbounded_String;
         end if;
      end Flush;
   begin
      --  Same pre-tokenization as training (inline, order-preserving).
      for I in Text'Range loop
         if Text (I) = ' ' then Flush; end if;
         Append (Cur, Text (I));
      end loop;
      Flush;

      return R : Id_Array (1 .. Natural (Out_V.Length)) do
         for I in R'Range loop
            R (I) := Out_V (Out_V.First_Index + (I - 1));
         end loop;
      end return;
   end Encode;

   function Decode (T : Trainer; Ids : Id_Array) return String is
      R : Unbounded_String;
   begin
      for Id of Ids loop Append (R, T.Pieces (Id)); end loop;
      return To_String (R);
   end Decode;

end BPE_Train;
