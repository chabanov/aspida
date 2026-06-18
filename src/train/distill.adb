---------------------------------------------------------------------
-- Distill body — top-K capture, teacher-distribution reconstruction, IO.
--
-- Real-teacher adapter (future, Phase 0->engine): wrap an LLM backend in a
-- type implementing Teacher, where Forward tokenizes/feeds the sequence and
-- copies the per-position output logits. That needs the backend to expose a
-- "forward -> logits" entry (the engines compute logits internally for the
-- sampler but don't surface them yet) — a small, localized addition.
---------------------------------------------------------------------

with Ada.Streams.Stream_IO;
with Ada.Numerics.Long_Elementary_Functions;
use  Ada.Numerics.Long_Elementary_Functions;

package body Distill is

   --------------------------------------------------------------------
   --  Top-K selection of one logit row (simple K-pass argmax; fine for the
   --  capture path — can become a heap when K and vocab grow).
   --------------------------------------------------------------------
   procedure Row_Top_K
     (L : Logit_Matrix; Row, K : Positive;
      Ids : out Id_Matrix; Vals : out Logit_Matrix)
   is
      V    : constant Positive := L'Length (2);
      Used : array (1 .. V) of Boolean := [others => False];
      Best : Positive;
      BestV : Logit;
      Found : Boolean;
   begin
      for Rank in 1 .. K loop
         Found := False;
         Best  := 1;
         BestV := Logit'First;
         for C in 1 .. V loop
            if not Used (C) and then (not Found or else L (Row, C) > BestV) then
               Best  := C;
               BestV := L (Row, C);
               Found := True;
            end if;
         end loop;
         Used (Best) := True;
         Ids  (Row, Rank) := Token (Best - 1);   -- 0-based token id
         Vals (Row, Rank) := BestV;
      end loop;
   end Row_Top_K;

   --------------------------------------------------------------------
   function Capture
     (T : in out Teacher'Class; Tokens : Token_Array; K : Positive)
      return Sample
   is
      N : constant Positive := Tokens'Length;
      L : Logit_Matrix (1 .. N, 1 .. Vocab (T));
      S : Sample (N, K);
   begin
      Forward (T, Tokens, L);
      for I in 1 .. N loop
         S.Tokens (I) := Tokens (Tokens'First + I - 1);
      end loop;
      for R in 1 .. N loop
         Row_Top_K (L, R, K, S.Top_Ids, S.Top_Logit);
      end loop;
      return S;
   end Capture;

   --------------------------------------------------------------------
   function Teacher_Prob
     (S : Sample; Row : Positive; Temp : Long_Float := 1.0) return Prob_Vector
   is
      P   : Prob_Vector (1 .. S.K);
      M   : Long_Float := Long_Float (S.Top_Logit (Row, 1));
      Sum : Long_Float := 0.0;
   begin
      for J in 2 .. S.K loop
         M := Long_Float'Max (M, Long_Float (S.Top_Logit (Row, J)));
      end loop;
      for J in 1 .. S.K loop
         P (J) := Exp ((Long_Float (S.Top_Logit (Row, J)) - M) / Temp);
         Sum := Sum + P (J);
      end loop;
      for J in 1 .. S.K loop
         P (J) := P (J) / Sum;
      end loop;
      return P;
   end Teacher_Prob;

   --------------------------------------------------------------------
   --  IO
   --------------------------------------------------------------------
   Magic : constant Natural := 16#AD571#;   -- "ADST" marker

   procedure Write (Path : String; Data : Dataset) is
      use Ada.Streams.Stream_IO;
      F : File_Type;
      S : Stream_Access;
   begin
      Create (F, Out_File, Path);
      S := Stream (F);
      Natural'Write (S, Magic);
      Natural'Write (S, Natural (Data.Length));
      for Smp of Data loop
         Natural'Write (S, Smp.N);
         Natural'Write (S, Smp.K);
         for I in 1 .. Smp.N loop
            Token'Write (S, Smp.Tokens (I));
         end loop;
         for I in 1 .. Smp.N loop
            for J in 1 .. Smp.K loop
               Token'Write (S, Smp.Top_Ids (I, J));
               Logit'Write (S, Smp.Top_Logit (I, J));
            end loop;
         end loop;
      end loop;
      Close (F);
   end Write;

   function Read (Path : String) return Dataset is
      use Ada.Streams.Stream_IO;
      F   : File_Type;
      S   : Stream_Access;
      D   : Dataset;
      Mg, Cnt, N, K : Natural;
   begin
      Open (F, In_File, Path);
      S := Stream (F);
      Natural'Read (S, Mg);
      if Mg /= Magic then
         Close (F);
         raise Constraint_Error with "not an Aspida distillation dataset";
      end if;
      Natural'Read (S, Cnt);
      for C in 1 .. Cnt loop
         Natural'Read (S, N);
         Natural'Read (S, K);
         declare
            Smp : Sample (N, K);
         begin
            for I in 1 .. N loop
               Token'Read (S, Smp.Tokens (I));
            end loop;
            for I in 1 .. N loop
               for J in 1 .. K loop
                  Token'Read (S, Smp.Top_Ids (I, J));
                  Logit'Read (S, Smp.Top_Logit (I, J));
               end loop;
            end loop;
            D.Append (Smp);
         end;
      end loop;
      Close (F);
      return D;
   end Read;

end Distill;
