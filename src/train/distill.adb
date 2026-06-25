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
with Ada.IO_Exceptions;
with Ada.Unchecked_Deallocation;
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
      --  Heap-allocate the [N x Vocab] logits: for large vocabularies (Gemma's
      --  262k) this is megabytes and must not sit on the primary stack.
      type LM_Ptr is access Logit_Matrix;
      procedure Free is new Ada.Unchecked_Deallocation (Logit_Matrix, LM_Ptr);
      L : LM_Ptr := new Logit_Matrix (1 .. N, 1 .. Vocab (T));
      S : Sample (N, K);
   begin
      Forward (T, Tokens, L.all);
      for I in 1 .. N loop
         S.Tokens (I) := Tokens (Tokens'First + I - 1);
      end loop;
      for R in 1 .. N loop
         Row_Top_K (L.all, R, K, S.Top_Ids, S.Top_Logit);
      end loop;
      Free (L);
      return S;
   exception
      when others =>
         Free (L);
         raise;
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
   --  Multi-teacher ensemble capture: average each teacher's temperature-
   --  scaled softmax (probability space), then keep the top-K of the blend.
   --------------------------------------------------------------------
   function Capture_Ensemble
     (Teachers : Teacher_Array;
      Tokens   : Token_Array;
      K        : Positive;
      Weights  : Weight_Array := [];
      Temp     : Long_Float   := 1.0)
      return Sample
   is
      N    : constant Positive := Tokens'Length;
      V    : constant Positive := Vocab (Teachers (Teachers'First).all);
      WSum : Long_Float := 0.0;
      S    : Sample (N, K);

      --  Heap-allocate the per-position full-vocab buffers: for large
      --  vocabularies (Gemma's 262k) the averaged distribution and a teacher's
      --  logits are tens of MB and must not sit on the primary stack.
      type Avg_Mat is array (1 .. N, 1 .. V) of Long_Float;
      type Avg_Ptr is access Avg_Mat;
      type LM_Ptr  is access Logit_Matrix;
      type Row_Vec is array (1 .. V) of Long_Float;
      type Row_Ptr is access Row_Vec;
      procedure Free is new Ada.Unchecked_Deallocation (Avg_Mat, Avg_Ptr);
      procedure Free is new Ada.Unchecked_Deallocation (Logit_Matrix, LM_Ptr);
      procedure Free is new Ada.Unchecked_Deallocation (Row_Vec, Row_Ptr);

      Avg : Avg_Ptr := new Avg_Mat;          -- weighted average distribution
      L   : LM_Ptr  := new Logit_Matrix (1 .. N, 1 .. V);  -- one teacher's logits
      P   : Row_Ptr := new Row_Vec;          -- a single row's softmax, reused

      function W_Of (Idx : Positive) return Long_Float is
        (if Weights'Length = 0 then 1.0
         else Weights (Weights'First + Idx - 1));

      procedure Free_All is
      begin
         Free (Avg);
         Free (L);
         Free (P);
      end Free_All;
   begin
      --  Every teacher must agree on the vocabulary; otherwise column c does
      --  not denote the same token across teachers and the average is garbage.
      for Ti in Teachers'Range loop
         if Vocab (Teachers (Ti).all) /= V then
            raise Vocab_Mismatch
              with "ensemble teachers must share one vocabulary";
         end if;
      end loop;

      for I in 1 .. Teachers'Length loop
         WSum := WSum + W_Of (I);
      end loop;
      if WSum <= 0.0 then
         raise Vocab_Mismatch with "ensemble weights must sum to > 0";
      end if;

      for R in 1 .. N loop
         for C in 1 .. V loop
            Avg (R, C) := 0.0;
         end loop;
      end loop;

      --  Accumulate each teacher's weighted, temperature-scaled softmax into
      --  the running average distribution.
      for Idx in 1 .. Teachers'Length loop
         declare
            T : constant Teacher_Ptr := Teachers (Teachers'First + Idx - 1);
            W : constant Long_Float  := W_Of (Idx) / WSum;
         begin
            Forward (T.all, Tokens, L.all);
            for R in 1 .. N loop
               declare
                  M   : Long_Float := Long_Float (L (R, 1));
                  Sum : Long_Float := 0.0;
               begin
                  for C in 2 .. V loop
                     M := Long_Float'Max (M, Long_Float (L (R, C)));
                  end loop;
                  for C in 1 .. V loop
                     P (C) := Exp ((Long_Float (L (R, C)) - M) / Temp);
                     Sum := Sum + P (C);
                  end loop;
                  for C in 1 .. V loop
                     Avg (R, C) := Avg (R, C) + W * (P (C) / Sum);
                  end loop;
               end;
            end loop;
         end;
      end loop;

      for I in 1 .. N loop
         S.Tokens (I) := Tokens (Tokens'First + I - 1);
      end loop;

      --  Top-K of the averaged distribution per position. Store log(prob) as
      --  the "logit" so Teacher_Prob's softmax reconstructs the (renormalized)
      --  ensemble top-K probabilities — identical downstream behaviour to a
      --  single-teacher Sample.
      for R in 1 .. N loop
         declare
            Used  : array (1 .. V) of Boolean := [others => False];
            Floor : constant Long_Float := 1.0e-30;
            Best  : Positive;
            BestV : Long_Float;
            Found : Boolean;
         begin
            for Rank in 1 .. K loop
               Found := False;
               Best  := 1;
               BestV := Long_Float'First;
               for C in 1 .. V loop
                  if not Used (C)
                    and then (not Found or else Avg (R, C) > BestV)
                  then
                     Best  := C;
                     BestV := Avg (R, C);
                     Found := True;
                  end if;
               end loop;
               Used (Best) := True;
               S.Top_Ids   (R, Rank) := Token (Best - 1);   -- 0-based id
               S.Top_Logit (R, Rank) :=
                 Logit (Log (Long_Float'Max (BestV, Floor)));
            end loop;
         end;
      end loop;

      Free_All;
      return S;
   exception
      when others =>
         Free_All;
         raise;
   end Capture_Ensemble;

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

      --  Minimum on-disk byte cost, used to bound an untrusted count against
      --  the bytes actually left in the file. Conservative: at least one
      --  stream element per item, so a declared count can never exceed what
      --  the file could possibly contain.
      Min_Sample_Bytes : constant Count := 8;   -- the N and K headers
      Min_Item_Bytes   : constant Count := 1;

      --  Reject obviously absurd dimensions outright (defence in depth on top
      --  of the file-size bound: a giant Sample (N, K) would otherwise try to
      --  allocate before the per-item loops are ever reached).
      Max_Dim : constant Natural := 2 ** 24;   -- 16M tokens / top-K width

      function Remaining return Count is
        (if Size (F) >= Index (F) then Size (F) - Index (F) + 1 else 0);
   begin
      Open (F, In_File, Path);
      S := Stream (F);
      Natural'Read (S, Mg);
      if Mg /= Magic then
         Close (F);
         raise Bad_Dataset with "not an Aspida distillation dataset";
      end if;

      Natural'Read (S, Cnt);
      --  Each remaining sample costs at least Min_Sample_Bytes on disk, so a
      --  count larger than the remaining file is corrupt — fail loud instead
      --  of looping billions of times.
      if Count (Cnt) > Remaining / Min_Sample_Bytes then
         Close (F);
         raise Bad_Dataset with "dataset sample count exceeds file size";
      end if;

      for C in 1 .. Cnt loop
         Natural'Read (S, N);
         Natural'Read (S, K);
         if N < 1 or else K < 1 or else N > Max_Dim or else K > Max_Dim then
            Close (F);
            raise Bad_Dataset with "dataset sample has out-of-range N/K";
         end if;
         --  N tokens + N*K (id, logit) items must fit in the remaining bytes.
         if Count (N) + Count (N) * Count (K) > Remaining / Min_Item_Bytes then
            Close (F);
            raise Bad_Dataset with "dataset sample exceeds remaining file size";
         end if;
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
   exception
      when Ada.IO_Exceptions.End_Error | Ada.IO_Exceptions.Data_Error =>
         if Is_Open (F) then Close (F); end if;
         raise Bad_Dataset with "truncated or corrupt distillation dataset";
   end Read;

end Distill;
