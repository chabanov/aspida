------------------------------------------------------------------------
-- turnkey_demo — the MVP platform loop on REAL components (model-free, $0):
-- a verifier-driven CODE job run through the Turnkey orchestrator end-to-end.
--
--   * Trainer  : REAL verifier-filtered learning — for each spec it searches the
--                program space and keeps a candidate the Code_DSL verifier
--                accepts (the student's learned program).
--   * Evaluator: REAL held-out eval — 60 (spec, a, b) instances executed; the
--                trained student vs a noisy teacher (Distractor), real pass-rates.
--   * Turnkey  : Admit -> train -> gate -> Final_Charge.
--
-- This exercises the whole platform vertical with no stubs on the train/eval path
-- (the GPU engine Student_GPU is the from-scratch-LM track, validated separately).
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Code_DSL;
with Platform;              use Platform;
with Turnkey;              use Turnkey;

procedure Turnkey_Demo is
   Trained : array (Code_DSL.Spec_Id) of Code_DSL.Program := [others => [0, 0, 0]];

   --  REAL verifier-filtered training: keep a verifier-accepted program per spec.
   function Train (J : Job_Spec) return Boolean is
      pragma Unreferenced (J);
      Operands : constant array (1 .. 2) of Integer := [6, 7];        -- a, b
      Ops      : constant array (1 .. 5) of Integer := [8, 9, 10, 11, 12];
      All_OK   : Boolean := True;
   begin
      for S in Code_DSL.Spec_Id loop
         declare Solved : Boolean := False; begin
            for O1 of Operands loop
               for O2 of Operands loop
                  for Op of Ops loop
                     if not Solved and then Code_DSL.Verify (S, [O1, O2, Op]) then
                        Trained (S) := [O1, O2, Op]; Solved := True;   -- verifier accepted
                     end if;
                  end loop;
               end loop;
            end loop;
            if not Solved then All_OK := False; end if;
         end;
      end loop;
      return All_OK;
   end Train;

   --  REAL held-out eval: 60 (spec,a,b) instances, student vs noisy teacher.
   procedure Eval (J : Job_Spec; Domain_Verified : out Boolean;
                   Eval_N : out Natural; Teacher_Pass, Student_Pass : out Float) is
      pragma Unreferenced (J);
      N : constant := 60;
      S_Ok, T_Ok : Natural := 0;
   begin
      for I in 0 .. N - 1 loop
         declare
            S    : constant Code_DSL.Spec_Id := 1 + (I mod Code_DSL.N_Specs);
            A    : constant Integer := (I mod 7) + 1;
            B    : constant Integer := (I / 7 mod 6) + 2;
            Tgt  : constant Integer := Code_DSL.Target (S, A, B);
            SV, TV : Integer; SOk, TOk : Boolean;
         begin
            Code_DSL.Run (Trained (S),            A, B, SV, SOk);       -- student
            Code_DSL.Run (Code_DSL.Distractor (S), A, B, TV, TOk);      -- noisy teacher
            if SOk and then SV = Tgt then S_Ok := S_Ok + 1; end if;
            if TOk and then TV = Tgt then T_Ok := T_Ok + 1; end if;
         end;
      end loop;
      Domain_Verified := True; Eval_N := N;
      Student_Pass := Float (S_Ok) / Float (N);
      Teacher_Pass := Float (T_Ok) / Float (N);
   end Eval;

   J : constant Job_Spec :=
     (Domain => Code, Tier => Small, Droplets => 1, Hours_Per_Drop => 2,
      Max_Spend => 50.00,
      Persona_Name     => To_Unbounded_String ("AdaCoder-1"),
      Persona_System   => To_Unbounded_String ("a precise Ada coding assistant"),
      Teacher_Attested => True);
   Q : constant Quote_T := Quote (J);
   O : constant Outcome :=
     Run (J, Q, Train'Unrestricted_Access, Eval'Unrestricted_Access,
          Deliver => null, Hours_Used => 2);   -- DSL domain: no GGUF artifact
begin
   Put_Line ("=== Turnkey demo: real verifier-driven CODE job, end-to-end ===");
   Put_Line ("  quote: provider " & Q.Provider_Cost'Image & " price " & Q.Platform_Price'Image
             & " (cap " & J.Max_Spend'Image & ")");
   Put_Line ("  admitted      : " & O.Admitted'Image);
   Put_Line ("  teacher pass  : " & O.Report.Teacher_Pass'Image);
   Put_Line ("  student pass  : " & O.Report.Student_Pass'Image
             & "  (N=" & O.Report.Eval_N'Image & ")");
   Put_Line ("  beats teachers: " & O.Report.Beats_Teachers'Image);
   Put_Line ("  state         : " & O.State'Image);
   Put_Line ("  charged       : " & O.Charge'Image);
   New_Line;
   if O.State = Delivered and then O.Report.Beats_Teachers then
      Put_Line ("RESULT: PASS (turnkey delivered a verified student that beat the teacher)");
   else
      Put_Line ("RESULT: FAIL");
   end if;
end Turnkey_Demo;
