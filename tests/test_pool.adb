---------------------------------------------------------------------
-- Test LLM_Pool — persistent worker pool
--
--   A. Parallel Run covers the whole range exactly once (each index
--      written by exactly one Execute, result matches serial).
--   B. Exception safety: an Op that raises in Execute surfaces as an
--      exception from Run AND leaves the pool usable (not wedged) — a
--      subsequent normal Run still completes. This is the regression
--      guard for the worker-deadlock / Pool_Busy-stuck bug.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Pool;

procedure Test_Pool is
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

   N : constant := 10_000;                    -- big enough to go parallel
   type Int_Array is array (1 .. N) of Integer;
   Buf : Int_Array := [others => 0];

   --  Writes 2*i into each covered index (disjoint slices, no locking).
   type Fill_Op is new LLM_Pool.Parallel_Op with null record;
   overriding procedure Execute (Op : in out Fill_Op; Lo, Hi : Integer) is
   begin
      for I in Lo .. Hi loop
         Buf (I) := 2 * I;
      end loop;
   end Execute;

   --  Raises partway through to exercise the exception path.
   type Boom_Op is new LLM_Pool.Parallel_Op with null record;
   overriding procedure Execute (Op : in out Boom_Op; Lo, Hi : Integer) is
   begin
      for I in Lo .. Hi loop
         if I mod 5_000 = 0 then
            raise Constraint_Error with "boom";
         end if;
      end loop;
   end Execute;

begin
   Put_Line ("=== LLM_Pool Test Suite ===");
   New_Line;

   --  A. Full, exactly-once coverage.
   declare
      F : Fill_Op;
      Ok : Boolean := True;
   begin
      LLM_Pool.Run (F, 1, N, Min_Grain => 256);
      for I in 1 .. N loop
         if Buf (I) /= 2 * I then
            Ok := False;
         end if;
      end loop;
      Assert ("parallel Run covers whole range exactly once", Ok);
   end;

   --  B1. An Op that raises surfaces as an exception from Run.
   declare
      B : Boom_Op;
      Raised : Boolean := False;
   begin
      LLM_Pool.Run (B, 1, N, Min_Grain => 256);
      Assert ("Run raises when Op.Execute raises", False);
   exception
      when others =>
         Raised := True;
         Assert ("Run raises when Op.Execute raises", Raised);
   end;

   --  B2. Pool still usable after a fault (not wedged in serial / deadlock).
   declare
      F : Fill_Op;
      Ok : Boolean := True;
   begin
      Buf := [others => 0];
      LLM_Pool.Run (F, 1, N, Min_Grain => 256);
      for I in 1 .. N loop
         if Buf (I) /= 2 * I then
            Ok := False;
         end if;
      end loop;
      Assert ("pool still works after an exception", Ok);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Pool;
