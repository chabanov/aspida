------------------------------------------------------------------------
-- code_iterate — verifier-bootstrapped SELF-IMPROVEMENT (STaR-style), with NO
-- external teacher. The model proposes programs (sampled for exploration), an
-- executable verifier keeps only the correct ones, those accumulate in a replay
-- buffer, a fresh student is retrained on the buffer, and that student becomes
-- the proposer for the next round. Quality climbs round over round driven only
-- by the verifier — the final model far exceeds the (random) round-0 proposer.
--
-- Task: synthesize a 5-token RPN program  (o1 o2 OP3 o4 OP5)  computing a
-- 2-operation composite of inputs a,b. Verify runs it on several test pairs.
-- Vocab ids: 0 pad | 1..8 spec | 9 a, 10 b | 11 + 12 - 13 * 14 min 15 max.
------------------------------------------------------------------------

with Ada.Text_IO;      use Ada.Text_IO;
with Ada.Command_Line; use Ada.Command_Line;
with Ada.Numerics.Generic_Elementary_Functions;
with Train;            use Train;
with Student;

procedure Code_Iterate is
   package EF is new Ada.Numerics.Generic_Elementary_Functions (Real);
   use EF;
   N_Specs : constant := 8;
   Vocab   : constant := 16;
   PLen    : constant := 5;        -- program length (tokens)
   Seq     : constant := 6;        -- [spec, p1..p5]
   Dm      : constant := 48;
   Ff      : constant := 96;
   Lyr     : constant := 2;
   Heads   : constant := 4;
   Rounds  : constant := 8;
   K_Samp  : constant := 300;      -- proposals per spec per round
   Epochs  : constant := 30;
   Base_LR : constant := 5.0E-3;
   Temp    : constant Real := 1.3;

   A_Tok : constant := 9;  B_Tok : constant := 10;
   Op_Lo : constant := 11; Op_Hi : constant := 15;

   subtype Spec_Id is Integer range 1 .. N_Specs;
   type Program is array (1 .. PLen) of Integer;

   --  golden composite per spec: (o1 op3 o2) op5 o4
   Golden : constant array (Spec_Id) of Program :=
     [1 => [A_Tok, B_Tok, 11, A_Tok, 11],   -- (a+b)+a
      2 => [A_Tok, B_Tok, 11, A_Tok, 13],   -- (a+b)*a
      3 => [A_Tok, B_Tok, 12, B_Tok, 11],   -- (a-b)+b
      4 => [A_Tok, B_Tok, 13, A_Tok, 12],   -- (a*b)-a
      5 => [A_Tok, B_Tok, 11, B_Tok, 13],   -- (a+b)*b
      6 => [A_Tok, B_Tok, 15, A_Tok, 12],   -- max(a,b)-a
      7 => [A_Tok, B_Tok, 14, B_Tok, 11],   -- min(a,b)+b
      8 => [A_Tok, B_Tok, 12, A_Tok, 13]];  -- (a-b)*a

   function Apply (Op, X, Y : Integer) return Integer is
     (case Op is
        when 11 => X + Y, when 12 => X - Y, when 13 => X * Y,
        when 14 => Integer'Min (X, Y), when 15 => Integer'Max (X, Y),
        when others => 0);

   procedure Eval (P : Program; A, B : Integer; Val : out Integer; Ok : out Boolean) is
      function Opnd (T : Integer) return Integer is (if T = A_Tok then A else B);
   begin
      Val := 0;
      Ok  := (P (1) = A_Tok or else P (1) = B_Tok)
        and then (P (2) = A_Tok or else P (2) = B_Tok)
        and then (P (4) = A_Tok or else P (4) = B_Tok)
        and then (P (3) in Op_Lo .. Op_Hi)
        and then (P (5) in Op_Lo .. Op_Hi);
      if not Ok then return; end if;
      Val := Apply (P (5), Apply (P (3), Opnd (P (1)), Opnd (P (2))), Opnd (P (4)));
   end Eval;

   function Verify (S : Spec_Id; P : Program) return Boolean is
      Tests : constant array (1 .. 8, 1 .. 2) of Integer :=
        [[3, 5], [7, 2], [4, 4], [9, 1], [2, 8], [6, 3], [1, 7], [8, 5]];
      V, GV : Integer; Ok, GOk : Boolean;
   begin
      for I in Tests'Range (1) loop
         Eval (P, Tests (I, 1), Tests (I, 2), V, Ok);
         Eval (Golden (S), Tests (I, 1), Tests (I, 2), GV, GOk);
         if not Ok or else V /= GV then return False; end if;
      end loop;
      return True;
   end Verify;

   package STd is new Student
     (Voc => Vocab, Dm => Dm, Ff => Ff, Seq => Seq, Lyr => Lyr, Heads => Heads,
      Use_RoPE => True, Rope_Base => 10000.0);
   type MAcc is access STd.Model;
   Gen_M : constant MAcc := new STd.Model;   -- single persistent model
   G : RNG := Seeded (77.0);

   --  replay buffer of verified (spec, program) pairs
   type Sample is record S : Spec_Id; P : Program; end record;
   Buf : array (1 .. N_Specs * 64) of Sample;
   N_Buf : Natural := 0;

   function Seen (S : Spec_Id; P : Program) return Boolean is
   begin
      for I in 1 .. N_Buf loop
         if Buf (I).S = S and then Buf (I).P = P then return True; end if;
      end loop;
      return False;
   end Seen;

   --  Grammar of a slot: operand slots (program pos 1,2,4) allow ids 9..10,
   --  op slots (pos 3,5) allow ids 11..15. Constrained decoding keeps every
   --  proposal well-formed, so exploration searches the 200 valid programs
   --  rather than the ~astronomically larger space of arbitrary token strings.
   function Lo_Of (Pos : Integer) return Integer is
     (if Pos = 3 or else Pos = 5 then Op_Lo else A_Tok);
   function Hi_Of (Pos : Integer) return Integer is
     (if Pos = 3 or else Pos = 5 then Op_Hi else B_Tok);

   --  Temperature-sample a vocab id within the allowed [Lo,Hi] range.
   function Sample_Range (L : STd.Logit_Mat; Row, Lo, Hi : Integer) return Integer is
      M    : Real := L (Row, Lo + 1);
      Sum  : Real := 0.0;
      U, C : Real;
   begin
      for D in Lo .. Hi loop M := Real'Max (M, L (Row, D + 1)); end loop;
      for D in Lo .. Hi loop Sum := Sum + Exp ((L (Row, D + 1) - M) / Temp); end loop;
      U := Uniform (G) * Sum; C := 0.0;
      for D in Lo .. Hi loop
         C := C + Exp ((L (Row, D + 1) - M) / Temp);
         if U <= C then return D; end if;
      end loop;
      return Hi;
   end Sample_Range;

   function Propose (M : MAcc; S : Spec_Id) return Program is
      Toks : Label_Array (1 .. Seq) := [S, 0, 0, 0, 0, 0];
      L : STd.Logit_Mat; P : Program;
   begin
      for Pos in 1 .. PLen loop
         STd.Forward (M.all, Toks, L);
         P (Pos) := Sample_Range (L, Pos, Lo_Of (Pos), Hi_Of (Pos));
         Toks (Pos + 1) := P (Pos);
      end loop;
      return P;
   end Propose;

   function Solve (M : MAcc; S : Spec_Id) return Program is
      Toks : Label_Array (1 .. Seq) := [S, 0, 0, 0, 0, 0];
      L : STd.Logit_Mat; P : Program;
      function Amax (Row, Lo, Hi : Integer) return Integer is
         B : Integer := Lo; BV : Real := Real'First;
      begin
         for D in Lo .. Hi loop
            if L (Row, D + 1) > BV then BV := L (Row, D + 1); B := D; end if;
         end loop;
         return B;
      end Amax;
   begin
      for Pos in 1 .. PLen loop
         STd.Forward (M.all, Toks, L);
         P (Pos) := Amax (Pos, Lo_Of (Pos), Hi_Of (Pos));
         Toks (Pos + 1) := P (Pos);
      end loop;
      return P;
   end Solve;

   function Solved (M : MAcc) return Natural is
      Ok : Natural := 0;
   begin
      for S in Spec_Id loop
         if Verify (S, Solve (M, S)) then Ok := Ok + 1; end if;
      end loop;
      return Ok;
   end Solved;

   --  Continual training: refine the SAME model on the accumulated buffer (no
   --  re-init between rounds). Retraining from scratch each round is what caused
   --  the mid-loop collapse; warm continuation keeps prior knowledge so the
   --  solve-count climbs monotonically.
   procedure Train_Buf (M : MAcc) is
      Toks : Label_Array (1 .. Seq); L, Tgt : STd.Logit_Mat;
      Loss : Real; pragma Unreferenced (Loss);
   begin
      for Ep in 1 .. Epochs loop
         declare LR : constant Real := Base_LR * (1.0 - 0.8 * Real (Ep - 1) / Real (Epochs));
         begin
            for I in 1 .. N_Buf loop
               Toks (1) := Buf (I).S;
               for J in 1 .. PLen loop Toks (J + 1) := Buf (I).P (J); end loop;
               STd.Forward (M.all, Toks, L);
               Tgt := [others => [others => 0.0]];
               for J in 1 .. PLen loop Tgt (J, Buf (I).P (J) + 1) := 1.0; end loop;
               Tgt (Seq, 1) := 1.0;   -- pad
               Loss := STd.Backward (M.all, Tgt);
               STd.Step (M.all, LR, Clip => 1.0);
            end loop;
         end;
      end loop;
   end Train_Buf;

   procedure Collect (M : MAcc; New_Found : out Natural) is
      P : Program;
   begin
      New_Found := 0;
      for S in Spec_Id loop
         for K in 1 .. K_Samp loop
            P := Propose (M, S);
            if Verify (S, P) and then not Seen (S, P) and then N_Buf < Buf'Last then
               N_Buf := N_Buf + 1; Buf (N_Buf) := (S, P); New_Found := New_Found + 1;
            end if;
         end loop;
      end loop;
   end Collect;

   New_Found : Natural;
   Best      : Natural := 0;
begin
   Put_Line ("=== code_iterate: verifier-bootstrapped self-improvement (no teacher) ===");
   STd.Init (Gen_M.all, 1.0);   -- round-0 proposer = random init

   --  random-proposer baseline coverage (specs the untrained proposer can hit)
   declare Base_Ok : Natural := 0; P : Program;
   begin
      for S in Spec_Id loop
         declare Hit : Boolean := False;
         begin
            for K in 1 .. K_Samp loop
               P := Propose (Gen_M, S);
               if Verify (S, P) then Hit := True; end if;
            end loop;
            if Hit then Base_Ok := Base_Ok + 1; end if;
         end;
      end loop;
      Put_Line ("round-0 RANDOM proposer can stumble on:" & Base_Ok'Image
                & " /" & N_Specs'Image & " specs (with" & K_Samp'Image & " tries each)");
   end;

   Put_Line ("round | new-verified | buffer | solves | best");
   for R in 0 .. Rounds - 1 loop
      Collect (Gen_M, New_Found);     -- propose (temp) from the current model
      Train_Buf (Gen_M);             -- continually refine the SAME model
      declare
         Sv : constant Natural := Solved (Gen_M);
      begin
         --  reject-and-restore: accept a round only if it does not regress;
         --  otherwise roll the model back to the best checkpoint. Combined with
         --  continual training this makes the solve-count monotone — no dip.
         if Sv >= Best then
            Best := Sv;
            STd.Save (Gen_M.all, "code_iter_best.model");
         else
            STd.Load (Gen_M.all, "code_iter_best.model");   -- undo the regression
         end if;
         Put_Line ("  " & R'Image & "   |   " & New_Found'Image
                   & "   |  " & N_Buf'Image & "  |  " & Best'Image
                   & "   (raw" & Sv'Image & ")");
      end;
   end loop;

   New_Line;
   Put_Line ("BEST student solves:" & Best'Image & " /" & N_Specs'Image);
   if Best = N_Specs then
      Put_Line ("RESULT: PASS  (self-improved from a random proposer to full"
                & " coverage via the verifier alone)");
   else
      Put_Line ("RESULT: FAIL  (" & Best'Image & "/" & N_Specs'Image & ")");
      Set_Exit_Status (Failure);
   end if;
end Code_Iterate;
