---------------------------------------------------------------------
-- LLM_Step_Lock — fine-grained serialization of ONE transformer step.
--
-- The GPU matvec shim (shared device buffers + weight-cache map) and the
-- LLM_Pool worker pool both require that only one forward step runs at a
-- time. Previously the server held a coarse lock for a whole generation,
-- so a second user's request waited for the first to finish completely.
--
-- This lock is acquired around each single forward step (one token) and
-- RELEASED between steps. Concurrent generations from different sessions
-- then interleave token-by-token (fair FIFO entry queue) — both users see
-- their answer stream at once, each at ~1/N speed, instead of one waiting
-- behind the other. Throughput is unchanged (one GPU); latency is fair.
---------------------------------------------------------------------

package LLM_Step_Lock is

   procedure Acquire;   --  blocks until no other step is running
   procedure Release;   --  must pair with Acquire (use the exception-safe
                        --  helpers in the backends, which release on any exit)

end LLM_Step_Lock;
