---------------------------------------------------------------------
-- LLM_Pool — persistent worker-thread pool for data-parallel ranges.
--
-- Unlike LLM_Parallel (which spawns and joins fresh Ada tasks on every
-- call), this keeps N-1 worker tasks alive for the whole program and
-- dispatches sub-ranges to them, so the hot inner loops (matvecs, the LM
-- head) pay no task create/destroy cost per call.
--
-- Work is supplied as a tagged "operation": derive from Parallel_Op,
-- override Execute to process a disjoint sub-range [Lo, Hi], then call
-- Run. Each Execute runs on a distinct sub-range so it may write its
-- slice without locking. The calling thread itself runs one chunk and
-- waits for the workers, so all cores stay busy.
--
-- Re-entrant calls (an Execute that itself calls Run) and calls smaller
-- than Min_Grain run serially in the caller, which both avoids worker
-- starvation under nested parallelism and skips overhead on tiny loops.
---------------------------------------------------------------------

package LLM_Pool is

   type Parallel_Op is abstract tagged limited null record;

   --  Process the disjoint sub-range [Lo, Hi]. Overridden per call site;
   --  the body typically closes over the enclosing kernel's locals.
   procedure Execute (Op : in out Parallel_Op; Lo, Hi : Integer) is abstract;

   --  Split [First, Last] across the pool and the calling thread.
   procedure Run
     (Op          : in out Parallel_Op'Class;
      First, Last : Integer;
      Min_Grain   : Integer := 256);

end LLM_Pool;
