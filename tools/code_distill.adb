------------------------------------------------------------------------
-- code_distill — verifier-filtered distillation: a student that BEATS a noisy
-- teacher on a narrow, executable-verified code-synthesis task.
--
-- A synthetic teacher proposes 3-token programs for each spec, but makes a
-- SYSTEMATIC error 60% of the time. We compare three things on the same task:
--   * the teacher's pass-rate (run through Code_DSL.Verify),
--   * a NAIVE student distilled from ALL teacher outputs (inherits the error),
--   * a FILTERED student distilled only from VERIFIED-correct outputs.
-- The filtered student exceeds the teacher because the executable verifier is
-- new information the teacher's distribution does not contain. The filtered
-- student is then exported to GGUF and re-served by the real engine, closing
-- teach -> verify -> train -> serve.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Command_Line;      use Ada.Command_Line;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Train;                 use Train;
with Student;
with Code_DSL;              use Code_DSL;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Code_Distill is
   Dm      : constant := 32;
   Ff      : constant := 64;
   Seq     : constant := 4;       -- [spec, op1, op2, op3]
   Lyr     : constant := 2;
   Heads   : constant := 2;
   Epochs  : constant := 40;
   Base_LR : constant := 5.0E-3;
   N_Train : constant := 2000;
   P_Correct : constant Real := 0.40;   -- teacher correct 40%, biased wrong 60%

   package STd is new Student
     (Voc => Vocab, Dm => Dm, Ff => Ff, Seq => Seq, Lyr => Lyr, Heads => Heads,
      Use_RoPE => True, Rope_Base => 10000.0);
   type MAcc is access STd.Model;
   M_Naive : constant MAcc := new STd.Model;
   M_Filt  : constant MAcc := new STd.Model;
   G : RNG := Seeded (123.0);

   function Rnd_Spec return Spec_Id is
     (1 + Integer (Real'Floor (Uniform (G) * Real (N_Specs))));

   --  Noisy teacher: correct with prob P_Correct, else its systematic mistake.
   function Teach (S : Spec_Id) return Program is
     (if Uniform (G) < P_Correct then Golden (S) else Distractor (S));

   --  The executable oracle, reached only through the pluggable interface so the
   --  pipeline is verifier-agnostic (a compiler oracle would drop in here).
   Vf : Code_DSL.DSL_Verifier;
   function Correct (S : Spec_Id; P : Program) return Boolean is
     (Vf.Is_Correct (S, [P (1), P (2), P (3)]));

   type Sample is record S : Spec_Id; P : Program; end record;
   type Sample_Arr is array (Positive range <>) of Sample;
   All_D  : Sample_Arr (1 .. N_Train);
   Filt_D : Sample_Arr (1 .. N_Train);
   N_Filt : Natural := 0;

   procedure Train_On (M : MAcc; D : Sample_Arr) is
      Toks : Label_Array (1 .. Seq);
      L, Tgt : STd.Logit_Mat;
      Loss : Real;
      pragma Unreferenced (Loss);
   begin
      STd.Init (M.all, 5.0);
      for Ep in 1 .. Epochs loop
         declare
            LR : constant Real :=
              Base_LR * (1.0 - 0.8 * Real (Ep - 1) / Real (Epochs));
         begin
            for I in D'Range loop
               Toks := [Spec_Token (D (I).S),
                        D (I).P (1), D (I).P (2), D (I).P (3)];
               STd.Forward (M.all, Toks, L);
               Tgt := [others => [others => 0.0]];
               Tgt (1, D (I).P (1) + 1) := 1.0;   -- predict op1 from spec
               Tgt (2, D (I).P (2) + 1) := 1.0;   -- predict op2
               Tgt (3, D (I).P (3) + 1) := 1.0;   -- predict op
               Tgt (4, 1) := 1.0;                 -- pad
               Loss := STd.Backward (M.all, Tgt);
               STd.Step (M.all, LR, Clip => 1.0);
            end loop;
         end;
      end loop;
   end Train_On;

   --  Greedily synthesize a program from the trained student.
   function Solve (M : MAcc; S : Spec_Id) return Program is
      Toks : Label_Array (1 .. Seq) := [Spec_Token (S), 0, 0, 0];
      L : STd.Logit_Mat;
      function Amax (Row : Integer) return Integer is
         B : Integer := 0; BV : Real := Real'First;
      begin
         for D in 1 .. Vocab loop
            if L (Row, D) > BV then BV := L (Row, D); B := D - 1; end if;
         end loop;
         return B;
      end Amax;
      P : Program;
   begin
      STd.Forward (M.all, Toks, L); P (1) := Amax (1); Toks (2) := P (1);
      STd.Forward (M.all, Toks, L); P (2) := Amax (2); Toks (3) := P (2);
      STd.Forward (M.all, Toks, L); P (3) := Amax (3);
      return P;
   end Solve;

   function Student_Solves (M : MAcc) return Natural is
      Ok : Natural := 0;
   begin
      for S in Spec_Id loop
         if Correct (S, Solve (M, S)) then Ok := Ok + 1; end if;
      end loop;
      return Ok;
   end Student_Solves;

   function Pct (N, D : Natural) return String is
     (Integer'Image (Integer (100.0 * Real (N) / Real (D))) & "%");
begin
   Put_Line ("=== code_distill: verifier-filtered distillation (student > teacher) ===");

   --  build datasets from the noisy teacher; filter by the executable verifier
   for I in 1 .. N_Train loop
      declare
         S : constant Spec_Id := Rnd_Spec;
         P : constant Program := Teach (S);
      begin
         All_D (I) := (S, P);
         if Correct (S, P) then
            N_Filt := N_Filt + 1;
            Filt_D (N_Filt) := (S, P);
         end if;
      end;
   end loop;
   Put_Line ("teacher samples:" & N_Train'Image
             & "   verified-correct:" & N_Filt'Image
             & "  (" & Pct (N_Filt, N_Train) & " )");

   --  teacher pass-rate on a fresh draw
   declare
      Ok : Natural := 0; K : constant := 2000;
   begin
      for I in 1 .. K loop
         declare S : constant Spec_Id := Rnd_Spec;
         begin if Correct (S, Teach (S)) then Ok := Ok + 1; end if; end;
      end loop;
      Put_Line ("TEACHER   pass-rate: " & Pct (Ok, K));
   end;

   Train_On (M_Naive, All_D);
   Train_On (M_Filt, Filt_D (1 .. N_Filt));

   Put_Line ("NAIVE    student solves: " & Student_Solves (M_Naive)'Image
             & " /" & N_Specs'Image & "   (" & Pct (Student_Solves (M_Naive), N_Specs) & " )");
   Put_Line ("FILTERED student solves: " & Student_Solves (M_Filt)'Image
             & " /" & N_Specs'Image & "   (" & Pct (Student_Solves (M_Filt), N_Specs) & " )");

   --  ---- export the filtered student + re-serve with the real engine ----
   declare
      Toks_S : GGUF_Write.Str_List (1 .. Vocab);
   begin
      Toks_S (1)  := To_Unbounded_String ("<pad>");
      Toks_S (2)  := To_Unbounded_String ("F1");
      Toks_S (3)  := To_Unbounded_String ("F2");
      Toks_S (4)  := To_Unbounded_String ("F3");
      Toks_S (5)  := To_Unbounded_String ("F4");
      Toks_S (6)  := To_Unbounded_String ("F5");
      Toks_S (7)  := To_Unbounded_String ("a");
      Toks_S (8)  := To_Unbounded_String ("b");
      Toks_S (9)  := To_Unbounded_String ("+");
      Toks_S (10) := To_Unbounded_String ("-");
      Toks_S (11) := To_Unbounded_String ("*");
      Toks_S (12) := To_Unbounded_String ("min");
      Toks_S (13) := To_Unbounded_String ("max");
      STd.Export_GGUF (M_Filt.all, "code.gguf", Toks_S, Ctx => Seq);
      Put_Line ("exported code.gguf");
   end;

   declare
      LM : constant LLM_Llama.Llama_Model := LLM_Llama.Load ("code.gguf");
      Vc : constant Integer := LLM_Llama.Vocab_Size (LM);

      function Solve_E (S : Spec_Id) return Program is
         Ids : LLM_Tokenizer.Token_Array (1 .. Seq) := [Spec_Token (S), 0, 0, 0];
         function Amax (Len, Row : Integer) return Integer is
            F : constant LLM_Llama.Logits_Flat :=
              LLM_Llama.Forward_Logits (LM, Ids (1 .. Len));
            B : Integer := 0; BV : Float := Float'First;
         begin
            for D in 0 .. Vc - 1 loop
               if F (Row * Vc + D) > BV then BV := F (Row * Vc + D); B := D; end if;
            end loop;
            return B;
         end Amax;
         P : Program;
      begin
         P (1) := Amax (1, 0); Ids (2) := P (1);
         P (2) := Amax (2, 1); Ids (3) := P (2);
         P (3) := Amax (3, 2);
         return P;
      end Solve_E;

      Served_Ok : Natural := 0;
      Faithful  : Boolean := True;
   begin
      for S in Spec_Id loop
         if Correct (S, Solve_E (S)) then Served_Ok := Served_Ok + 1; end if;
         if Solve_E (S) /= Solve (M_Filt, S) then Faithful := False; end if;
      end loop;
      Put_Line ("ENGINE-served filtered student solves: " & Served_Ok'Image
                & " /" & N_Specs'Image);
      Put_Line ("served == trained: " & Faithful'Image);

      New_Line;
      --  Core claim: the filtered student must (a) solve more than the naive
      --  student, (b) reach full coverage, (c) survive export+serve bit-faithfully.
      if Student_Solves (M_Filt) > Student_Solves (M_Naive)
        and then Student_Solves (M_Filt) = N_Specs
        and then Served_Ok = N_Specs
        and then Faithful
      then
         Put_Line ("RESULT: PASS  (verifier-filtered student beats the noisy"
                   & " teacher; loop closed)");
      else
         Put_Line ("RESULT: FAIL  (see counts above)");
         Set_Exit_Status (Failure);
      end if;
   end;
end Code_Distill;
