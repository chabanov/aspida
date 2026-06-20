------------------------------------------------------------------------
-- test_qat_train — end-to-end QAT demonstration. Two tiny models learn the
-- successor function t -> (t+1) mod V:
--   * FP : trained in full precision,
--   * QAT: trained with 2-bit fake-quantized weights (straight-through).
-- Then both are evaluated under REAL 2-bit quantization. The FP model, quantized
-- post-hoc, degrades; the QAT model stays accurate -- which is the point of QAT.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;       use Train;
with Student;

procedure Test_QAT_Train is
   Voc   : constant := 8;
   Bits  : constant := 2;         -- 4 levels (0, ±amax): aggressive enough to expose QAT's value
   Steps : constant := 10000;

   --  FP forward (master weights), and the same architecture with 2-bit QAT.
   package S_FP  is new Student (Voc => Voc, Dm => 16, Ff => 32, Seq => 3, Lyr => 1,
                                 Heads => 2);
   package S_QAT is new Student (Voc => Voc, Dm => 16, Ff => 32, Seq => 3, Lyr => 1,
                                 Heads => 2, Use_QAT => True, QAT_Bits => Bits);

   G : RNG := Seeded (11.0);
   function Rnd (N : Integer) return Integer is
     (Integer (Real'Floor (Uniform (G) * Real (N))));

   --  Train one model on t -> (t+1) mod Voc (row 1 predicts the successor).
   generic
      with package SP is new Student (<>);
   procedure Train_Model (M : in out SP.Model);

   procedure Train_Model (M : in out SP.Model) is
      Toks : Label_Array (1 .. 3);
      L    : SP.Logit_Mat;
      P    : Matrix (1 .. 3, 1 .. Voc);
      Tgt  : SP.Logit_Mat;
      Loss : Real := 0.0;
      pragma Unreferenced (Loss);
   begin
      SP.Init (M, 5.0);
      for Step in 1 .. Steps loop
         declare T : constant Integer := Rnd (Voc); begin
            Toks := [T, 0, 0];
            SP.Forward (M, Toks, L);
            Softmax_Rows (L, P);
            Tgt := P;
            for C in 1 .. Voc loop Tgt (1, C) := 0.0; end loop;
            Tgt (1, (T + 1) mod Voc + 1) := 1.0;     -- one-hot successor
            Loss := SP.Backward (M, Tgt);
            SP.Step (M, 5.0e-3, Clip => 1.0);
         end;
      end loop;
   end Train_Model;

   procedure Train_FP  is new Train_Model (S_FP);
   procedure Train_QAT is new Train_Model (S_QAT);

   --  Accuracy of M's forward over all t (argmax of row 1).
   generic
      with package SP is new Student (<>);
   function Accuracy (M : in out SP.Model) return Integer;

   function Accuracy (M : in out SP.Model) return Integer is
      Toks : Label_Array (1 .. 3);
      L    : SP.Logit_Mat;
      Ok   : Integer := 0;
   begin
      for T in 0 .. Voc - 1 loop
         Toks := [T, 0, 0];
         SP.Forward (M, Toks, L);
         declare
            Best : Integer := 0; BV : Real := Real'First;
         begin
            for C in 0 .. Voc - 1 loop
               if L (1, C + 1) > BV then BV := L (1, C + 1); Best := C; end if;
            end loop;
            if Best = (T + 1) mod Voc then Ok := Ok + 1; end if;
         end;
      end loop;
      return Ok;
   end Accuracy;

   function Acc_FP  is new Accuracy (S_FP);
   function Acc_QAT is new Accuracy (S_QAT);

   FP  : S_FP.Model;
   QAT : S_QAT.Model;
   PQ  : S_QAT.Model;          -- FP weights run through 4-bit (post-hoc quant)
   Pass : Boolean := True;
   File : constant String := "/tmp/aspida_qat_fp.model";
begin
   Put_Line ("=== QAT end-to-end demonstration ("
             & Integer'Image (Bits) & "-bit weights) ===");

   Train_FP  (FP);
   Train_QAT (QAT);

   --  Move the FP-trained weights into a QAT (4-bit forward) instance to model
   --  naive post-hoc quantization of an FP-trained model.
   S_FP.Save (FP, File);
   S_QAT.Init (PQ, 1.0);
   S_QAT.Load (PQ, File);

   declare
      A_FP  : constant Integer := Acc_FP  (FP);    -- full precision
      A_QAT : constant Integer := Acc_QAT (QAT);   -- QAT-trained, 4-bit forward
      A_PQ  : constant Integer := Acc_QAT (PQ);    -- FP weights, 4-bit forward
   begin
      Put_Line ("  full-precision (FP)          :" & A_FP'Image  & " /" & Voc'Image);
      Put_Line ("  post-hoc quant (FP weights)  :" & A_PQ'Image  & " /" & Voc'Image);
      Put_Line ("  QAT (QAT-trained weights)    :" & A_QAT'Image & " /" & Voc'Image);

      if A_FP < Voc - 1 then
         Put_Line ("  FAIL: FP model did not learn the task"); Pass := False;
      end if;
      if A_QAT < A_PQ then
         Put_Line ("  FAIL: QAT not more robust than post-hoc quant"); Pass := False;
      end if;
      if A_QAT < Voc - 1 then
         Put_Line ("  FAIL: QAT model not accurate under 4-bit"); Pass := False;
      end if;
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_QAT_Train;
