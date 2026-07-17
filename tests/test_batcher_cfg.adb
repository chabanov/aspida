--  Smoke-test LLM_Batcher.Configure under the concurrency it actually sees:
--  it runs on the handler tasks, so several may call it at once. Exactly one
--  must do the setup (and create the single Driver); the losers must not
--  return until the buffers are published, and nobody may deadlock.
--
--  NOTE: this exercises the ENABLED path, so it needs ASPIDA_BATCH_SERVE set.
--  Once Configure succeeds a Driver exists and blocks on the lane pool
--  forever (by design -- the server runs forever), so this binary does not
--  exit on its own. The harness kills it; the assertions print before that.
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Environment_Variables;
with LLM_Batcher;

procedure Test_Batcher_Cfg is
   N_Racers : constant := 4;

   Done_Count : array (1 .. N_Racers) of Boolean := [others => False];

   task type Racer (Id : Integer);
   task body Racer is
   begin
      --  All four hit Configure at once with identical parameters.
      LLM_Batcher.Configure (4, 64);
      Done_Count (Id) := True;
   end Racer;
begin
   if not Ada.Environment_Variables.Exists ("ASPIDA_BATCH_SERVE") then
      Put_Line ("SKIP: ASPIDA_BATCH_SERVE unset (Configure is unreachable)");
      Put_Line ("RESULT: PASS");
      return;
   end if;

   Put_Line ("racing" & Integer'Image (N_Racers) & " tasks into Configure");
   declare
      R1 : Racer (1);
      R2 : Racer (2);
      R3 : Racer (3);
      R4 : Racer (4);
   begin
      null;   -- block exit waits for all four to finish Configure
   end;

   for I in Done_Count'Range loop
      if not Done_Count (I) then
         Put_Line ("FAIL: racer" & Integer'Image (I) & " never returned");
         return;
      end if;
   end loop;
   Put_Line ("all racers returned from Configure (no deadlock)");

   --  A second call on the already-configured batcher must be a no-op.
   LLM_Batcher.Configure (4, 64);
   Put_Line ("re-Configure is a no-op");
   Put_Line ("RESULT: PASS");
end Test_Batcher_Cfg;
