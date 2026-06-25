------------------------------------------------------------------------
-- test_multi_teacher — several DIFFERENT teachers jointly teach one student.
-- Validates Distill.Capture_Ensemble: probability-space averaging of the
-- teachers, top-K of the *blend*, optional weighting, the shared-vocab guard,
-- and an end-to-end Student convergence against the ensemble target. Because
-- the merged result is an ordinary Sample, the existing KL-training pipeline
-- consumes it unchanged — which is the whole point of the design.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;
with Student;
with Distill;

procedure Test_Multi_Teacher is
   use type Distill.Token;

   V     : constant := 32;     -- shared vocabulary
   D     : constant := 8;      -- model dim
   Ff    : constant := 16;     -- ffn dim
   Tn    : constant := 4;      -- sequence length
   K     : constant := 6;      -- ensemble top-K
   M     : constant := 6;      -- training sequences
   Steps : constant := 900;

   Pass : Boolean := True;
   procedure Chk (Name : String; Cond : Boolean) is
   begin
      Put_Line ("  " & Name & ": " & (if Cond then "OK" else "FAIL"));
      if not Cond then Pass := False; end if;
   end Chk;

   --  A peaked synthetic teacher whose argmax depends on Mode, so two
   --  instances with different Mode genuinely disagree on every position.
   type Synth is new Distill.Teacher with record
      Mode : Integer := 1;
   end record;
   overriding function Vocab (T : Synth) return Positive is (V);
   overriding procedure Forward
     (T : in out Synth; Tokens : Distill.Token_Array;
      Out_Logits : out Distill.Logit_Matrix)
   is
      Peak : Integer;
   begin
      for R in 1 .. Tokens'Length loop
         Peak := (Integer (Tokens (Tokens'First + R - 1)) * T.Mode) mod V + 1;
         for C in 1 .. V loop
            Out_Logits (R, C) := Distill.Logit (-0.4 * Float (abs (C - Peak)));
         end loop;
      end loop;
   end Forward;

   --  A teacher with a different vocabulary, to exercise the mismatch guard.
   type Synth_BadV is new Distill.Teacher with null record;
   overriding function Vocab (T : Synth_BadV) return Positive is (V + 1);
   overriding procedure Forward
     (T : in out Synth_BadV; Tokens : Distill.Token_Array;
      Out_Logits : out Distill.Logit_Matrix) is
   begin
      Out_Logits := [others => [others => 0.0]];
   end Forward;

   Ta, Tb : aliased Synth;
   Tbad   : aliased Synth_BadV;

   --  Swallow a Sample result so a call can be discarded without binding it.
   procedure Discard (S : Distill.Sample) is
   begin
      pragma Unreferenced (S);
      null;
   end Discard;

   Seq1 : Distill.Token_Array (1 .. Tn);

   --  Student instantiated at the shared vocabulary.
   package St is new Student
     (Voc => V, Dm => D, Ff => Ff, Seq => Tn, Lyr => 2, Heads => 1);
   Mdl : St.Model;
begin
   Put_Line ("=== multi-teacher (ensemble) distillation ===");
   Ta.Mode := 1;
   Tb.Mode := 3;
   for I in 1 .. Tn loop
      Seq1 (I) := Distill.Token ((I * 7) mod V);
   end loop;

   --  (1) Shared-vocabulary guard: mixing a V+1 teacher must fail loud.
   declare
      Raised : Boolean := False;
   begin
      begin
         Discard (Distill.Capture_Ensemble ([Ta'Unchecked_Access, Tbad'Unchecked_Access], Seq1, K));
      exception
         when Distill.Vocab_Mismatch => Raised := True;
      end;
      Chk ("mismatched vocab raises Vocab_Mismatch", Raised);
   end;

   --  (2) Weighting: all weight on Tb must reproduce Tb taught alone.
   declare
      Solo  : constant Distill.Sample := Distill.Capture (Tb, Seq1, K);
      Ens   : constant Distill.Sample :=
        Distill.Capture_Ensemble ([Ta'Unchecked_Access, Tb'Unchecked_Access], Seq1, K,
                                  Weights => [0.0, 1.0]);
      Psolo : constant Distill.Prob_Vector := Distill.Teacher_Prob (Solo, 1);
      Pens  : constant Distill.Prob_Vector := Distill.Teacher_Prob (Ens, 1);
      Match : Boolean := Solo.Top_Ids (1, 1) = Ens.Top_Ids (1, 1);
   begin
      for J in 1 .. K loop
         if abs (Psolo (J) - Pens (J)) > 1.0e-6 then Match := False; end if;
      end loop;
      Chk ("weight=1 on one teacher reproduces it alone", Match);
   end;

   --  (3) Blend: the uniform ensemble's top-K must contain BOTH teachers'
   --  argmax ids — each model genuinely contributes to the target.
   declare
      Tok1   : constant Integer := Integer (Seq1 (1));
      Pa     : constant Integer := (Tok1 * Ta.Mode) mod V;   -- Ta argmax (0-based)
      Pb     : constant Integer := (Tok1 * Tb.Mode) mod V;   -- Tb argmax (0-based)
      Ens    : constant Distill.Sample :=
        Distill.Capture_Ensemble ([Ta'Unchecked_Access, Tb'Unchecked_Access], Seq1, K);
      Has_A, Has_B : Boolean := False;
   begin
      for J in 1 .. K loop
         if Integer (Ens.Top_Ids (1, J)) = Pa then Has_A := True; end if;
         if Integer (Ens.Top_Ids (1, J)) = Pb then Has_B := True; end if;
      end loop;
      Chk ("ensemble top-K covers both teachers' peaks",
           Pa /= Pb and then Has_A and then Has_B);
   end;

   --  (4) End-to-end: build a multi-teacher dataset and train the student.
   declare
      Toks : array (1 .. M) of Train.Label_Array (1 .. Tn);
      Tgt  : array (1 .. M) of St.Logit_Mat;
      First_Loss, Last_Loss : Train.Real := 0.0;
   begin
      for Mi in 1 .. M loop
         declare
            DT : Distill.Token_Array (1 .. Tn);
         begin
            for T in 1 .. Tn loop
               Toks (Mi) (T) := (Mi * 5 + T * 3) mod V;
               DT (T) := Distill.Token (Toks (Mi) (T));
            end loop;
            declare
               S : constant Distill.Sample :=
                 Distill.Capture_Ensemble ([Ta'Unchecked_Access, Tb'Unchecked_Access], DT, K);
            begin
               Tgt (Mi) := [others => [others => 0.0]];
               for R in 1 .. Tn loop
                  declare
                     P : constant Distill.Prob_Vector :=
                       Distill.Teacher_Prob (S, R);
                  begin
                     for J in 1 .. K loop
                        Tgt (Mi) (R, Integer (S.Top_Ids (R, J)) + 1) :=
                          Train.Real (P (J));
                     end loop;
                  end;
               end loop;
            end;
         end;
      end loop;

      St.Init (Mdl, 5.0);
      for Step in 1 .. Steps loop
         declare
            Mi : constant Integer := (Step - 1) mod M + 1;
            Lg : St.Logit_Mat;
            L  : Train.Real;
         begin
            St.Forward (Mdl, Toks (Mi), Lg);
            L := St.Backward (Mdl, Tgt (Mi));
            St.Step (Mdl, LR => 5.0E-3);
            if Step <= M then
               First_Loss := First_Loss + L / Train.Real (M);
            end if;
            if Step > Steps - M then
               Last_Loss := Last_Loss + L / Train.Real (M);
            end if;
         end;
      end loop;

      Put_Line ("  first-epoch KL:" & First_Loss'Image
                & "   final KL:" & Last_Loss'Image);
      Chk ("student converged toward the ensemble",
           Last_Loss < First_Loss * 0.5);
   end;

   --  (5) Per-teacher weighted KL is equivalent to ensemble KL. The gradient
   --  of sum_t w_t KL(P_t || Q) equals the gradient of KL(P_avg || Q) with
   --  P_avg = sum_t w_t P_t (the losses differ only by a Q-independent
   --  constant). So a dedicated "per-teacher KL" training mode would produce
   --  identical updates to training against the ensemble target that
   --  Capture_Ensemble already builds — verified here at the gradient level.
   declare
      Vv   : constant := 8;
      Rows : constant := 3;
      QL              : Train.Matrix (1 .. Rows, 1 .. Vv);  -- student logits
      L1, L2          : Train.Matrix (1 .. Rows, 1 .. Vv);
      P1, P2, Pavg    : Train.Matrix (1 .. Rows, 1 .. Vv);
      D1, D2, Davg    : Train.Matrix (1 .. Rows, 1 .. Vv);
      Rg      : Train.RNG := Train.Seeded (3.0);
      Max_Err : Train.Real := 0.0;
   begin
      Train.Init_Glorot (QL, Rg);
      Train.Init_Glorot (L1, Rg);
      Train.Init_Glorot (L2, Rg);
      Train.Softmax_Rows (L1, P1);
      Train.Softmax_Rows (L2, P2);
      for R in 1 .. Rows loop
         for C in 1 .. Vv loop
            Pavg (R, C) := 0.5 * (P1 (R, C) + P2 (R, C));
         end loop;
      end loop;
      Train.KL_Backward (QL, P1,   D1);
      Train.KL_Backward (QL, P2,   D2);
      Train.KL_Backward (QL, Pavg, Davg);
      for R in 1 .. Rows loop
         for C in 1 .. Vv loop
            Max_Err := Train.Real'Max
              (Max_Err, abs (Davg (R, C) - 0.5 * (D1 (R, C) + D2 (R, C))));
         end loop;
      end loop;
      Chk ("per-teacher weighted KL == ensemble KL at gradient level",
           Max_Err < 1.0e-9);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Multi_Teacher;
