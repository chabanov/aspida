---------------------------------------------------------------------
-- LLM_Pool body
---------------------------------------------------------------------

with System.Multiprocessors;

package body LLM_Pool is

   type Op_Access is access all Parallel_Op'Class;

   N_CPU     : constant Integer :=
     Integer (System.Multiprocessors.Number_Of_CPUs);
   --  Persistent helpers; the calling thread runs one chunk too, so the
   --  effective parallelism is N_Workers + 1 = N_CPU.
   N_Workers : constant Integer := Integer'Max (0, N_CPU - 1);

   --  Only the (single) top-level caller dispatches to the pool; any call
   --  made while the pool is already busy (nested matvec inside a parallel
   --  expert, or a worker re-entering Run) falls back to serial execution.
   --
   --  INVARIANT: at most one task initiates a top-level Run at a time. The
   --  engine's generation loop is single-threaded above the pool, so this
   --  holds; Pool_Busy then needs no locking (workers and nested calls only
   --  read it). If you ever drive Run from multiple threads concurrently,
   --  add a mutex around the busy flag.
   Pool_Busy : Boolean := False with Atomic;

   --  Set by a worker (or the caller's own chunk) if its Op.Execute raised, so
   --  Run can surface the failure instead of returning a partial result.
   Worker_Failed : Boolean := False with Atomic;

   --  Counts outstanding worker chunks for the current dispatch.
   protected Barrier is
      procedure Arm (Count : Natural);
      procedure Signal;
      entry Wait;
   private
      Remaining : Natural := 0;
   end Barrier;

   protected body Barrier is
      procedure Arm (Count : Natural) is
      begin
         Remaining := Count;
      end Arm;

      procedure Signal is
      begin
         if Remaining > 0 then
            Remaining := Remaining - 1;
         end if;
      end Signal;

      entry Wait when Remaining = 0 is
      begin
         null;
      end Wait;
   end Barrier;

   task type Worker is
      entry Assign (Op : Op_Access; Lo, Hi : Integer);
   end Worker;

   task body Worker is
      My_Op : Op_Access;
      L, H  : Integer;
   begin
      loop
         select
            accept Assign (Op : Op_Access; Lo, Hi : Integer) do
               My_Op := Op;  L := Lo;  H := Hi;
            end Assign;
         or
            terminate;
         end select;
         --  A worker must ALWAYS Signal (else Barrier.Wait blocks forever) and
         --  must survive (else it stops serving future work). Record the fault
         --  so Run can raise on the caller side.
         begin
            My_Op.Execute (L, H);
         exception
            when others =>
               Worker_Failed := True;
         end;
         Barrier.Signal;
      end loop;
   end Worker;

   Workers : array (1 .. Integer'Max (1, N_Workers)) of Worker;

   procedure Run
     (Op          : in out Parallel_Op'Class;
      First, Last : Integer;
      Min_Grain   : Integer := 256)
   is
      Count : constant Integer := Last - First + 1;
   begin
      if Count <= 0 then
         return;
      end if;

      --  Serial fallback: tiny range, no helpers, or nested/re-entrant call.
      if Count < Min_Grain or else N_Workers = 0 or else Pool_Busy then
         Op.Execute (First, Last);
         return;
      end if;

      Worker_Failed := False;
      Pool_Busy := True;
      declare
         NN     : constant Integer := Integer'Min (N_Workers + 1, Count);
         Chunk  : constant Integer := (Count + NN - 1) / NN;
         Acc    : constant Op_Access := Op'Unchecked_Access;
         Posted : Natural := 0;
         Caller_Failed : Boolean := False;
      begin
         --  Dispatch chunks 2 .. NN to the workers; run chunk 1 here.
         for I in 2 .. NN loop
            declare
               Lo : constant Integer := First + (I - 1) * Chunk;
            begin
               exit when Lo > Last;
               Posted := Posted + 1;
            end;
         end loop;

         Barrier.Arm (Posted);

         for I in 2 .. NN loop
            declare
               Lo : constant Integer := First + (I - 1) * Chunk;
               Hi : constant Integer := Integer'Min (First + I * Chunk - 1, Last);
            begin
               exit when Lo > Last;
               Workers (I - 1).Assign (Acc, Lo, Hi);
            end;
         end loop;

         --  Calling thread takes chunk 1. Capture (don't propagate) a fault
         --  here so we still drain the barrier — the workers are running and
         --  WILL Signal; leaving without waiting would let them write into a
         --  result the caller has already abandoned.
         begin
            Op.Execute (First, Integer'Min (First + Chunk - 1, Last));
         exception
            when others =>
               Caller_Failed := True;
         end;

         Barrier.Wait;
         Pool_Busy := False;

         if Caller_Failed or else Worker_Failed then
            Worker_Failed := False;
            raise Program_Error
              with "LLM_Pool: a parallel operation raised an exception";
         end if;
      end;
      return;
   end Run;

end LLM_Pool;
