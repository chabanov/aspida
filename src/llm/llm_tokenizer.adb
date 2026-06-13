---------------------------------------------------------------------
-- LLM_Tokenizer body — byte-level BPE (greedy merge by rank)
---------------------------------------------------------------------

with Ada.Containers.Indefinite_Ordered_Maps;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

package body LLM_Tokenizer is

   NUL : constant Character := Character'Val (0);

   package Str_Int_Maps is new Ada.Containers.Indefinite_Ordered_Maps
     (Key_Type => String, Element_Type => Integer);
   package Int_Str_Maps is new Ada.Containers.Indefinite_Ordered_Maps
     (Key_Type => Integer, Element_Type => String);

   type Tokenizer_Data is record
      Vocab  : Str_Int_Maps.Map;   -- piece -> id
      Id2Tok : Int_Str_Maps.Map;   -- id -> piece
      Merges : Str_Int_Maps.Map;   -- "left<NUL>right" -> rank
      Loaded : Boolean := False;
   end record;

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

   --------------------------------------------------------------------
   -- Decode
   --------------------------------------------------------------------

   function Decode_One (T : Tokenizer; Id : Integer) return String is
   begin
      if T /= null and then T.Id2Tok.Contains (Id) then
         return T.Id2Tok.Element (Id);
      elsif Id in 0 .. 255 then
         return R : String (1 .. 1) do
            R (1) := Character'Val (Id);
         end return;
      else
         return "";
      end if;
   end Decode_One;

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
   begin
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

      -- Greedy byte-pair encoding by merge rank.
      declare
         Pieces : array (1 .. Text'Length) of Unbounded_String;
         N      : Natural := Text'Length;
      begin
         for I in 1 .. Text'Length loop
            --  length-1 slice avoids an array aggregate
            Pieces (I) :=
              To_Unbounded_String (Text (Text'First + I - 1 .. Text'First + I - 1));
         end loop;

         loop
            declare
               Best_Rank : Integer := Integer'Last;
               Best_I    : Natural := 0;
            begin
               for I in 1 .. N - 1 loop
                  declare
                     Key : constant String :=
                       To_String (Pieces (I)) & NUL & To_String (Pieces (I + 1));
                  begin
                     if T.Merges.Contains (Key)
                       and then T.Merges.Element (Key) < Best_Rank
                     then
                        Best_Rank := T.Merges.Element (Key);
                        Best_I := I;
                     end if;
                  end;
               end loop;

               exit when Best_I = 0;

               Pieces (Best_I) := Pieces (Best_I) & Pieces (Best_I + 1);
               for J in Best_I + 1 .. N - 1 loop
                  Pieces (J) := Pieces (J + 1);
               end loop;
               N := N - 1;
            end;
         end loop;

         return R : Token_Array (1 .. N) do
            for I in 1 .. N loop
               declare
                  P : constant String := To_String (Pieces (I));
               begin
                  if T.Vocab.Contains (P) then
                     R (I) := T.Vocab.Element (P);
                  elsif P'Length = 1 then
                     R (I) := Character'Pos (P (P'First));  -- per-byte fallback
                  else
                     R (I) := 0;  -- unknown piece
                  end if;
               end;
            end loop;
         end return;
      end;
   end Encode;

end LLM_Tokenizer;
