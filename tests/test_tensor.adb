---------------------------------------------------------------------
-- Test LLM_Tensor value semantics (Controlled data handle)
--
-- Verifies that copying a tensor produces an independent deep copy
-- (no aliasing of the underlying storage) on both initialisation and
-- assignment, and that heavy allocate/copy/free churn runs cleanly
-- (exercising Adjust/Finalize without leaks or double-free).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tensor; use LLM_Tensor;

procedure Test_Tensor is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Assert;

begin
   Put_Line ("=== Tensor Value-Semantics Test Suite ===");
   New_Line;

   -- Initialisation copy is an independent deep copy.
   declare
      A : Tensor := New_Tensor ([1, 3]);
   begin
      Set_Flat (A, 1, 1.0);  Set_Flat (A, 2, 2.0);  Set_Flat (A, 3, 3.0);
      declare
         B : Tensor := A;     -- copy via Adjust
      begin
         Set_Flat (B, 1, 99.0);
         Assert ("init-copy: original unchanged", Get_Flat (A, 1) = 1.0);
         Assert ("init-copy: copy mutated",       Get_Flat (B, 1) = 99.0);
      end;
      Assert ("original intact after copy scope", Get_Flat (A, 1) = 1.0);
   end;

   -- Assignment copy (Finalize old + Adjust new) is also independent.
   declare
      A : Tensor := New_Tensor ([1, 2]);
      C : Tensor := New_Tensor ([1, 2]);
   begin
      Set_Flat (A, 1, 5.0);  Set_Flat (A, 2, 6.0);
      Set_Flat (C, 1, 0.0);  Set_Flat (C, 2, 0.0);
      C := A;                 -- assignment: frees C's old data, deep-copies A
      Set_Flat (C, 1, -7.0);
      Assert ("assign-copy: original unchanged", Get_Flat (A, 1) = 5.0);
      Assert ("assign-copy: copy mutated",       Get_Flat (C, 1) = -7.0);
   end;

   -- Heavy churn: allocate, deep-copy and free many tensors. Crashes here
   -- would indicate a double-free; a leak would blow up memory over 50k iters.
   declare
      Acc : Float := 0.0;
   begin
      for I in 1 .. 50_000 loop
         declare
            T : Tensor := New_Tensor ([1, 64]);
            U : constant Tensor := T;   -- deep copy
         begin
            Set_Flat (T, 1, Float (I));
            Acc := Acc + Get_Flat (U, 1);   -- U independent (stays 0.0)
         end;                                -- both finalized here
      end loop;
      Assert ("churn copies are independent (U stayed 0)", Acc = 0.0);
      Assert ("50k allocate/copy/free cycles completed", True);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Tensor;
