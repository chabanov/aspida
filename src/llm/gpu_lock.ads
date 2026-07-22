------------------------------------------------------------------------------
--  GPU_Lock — cross-process reader/writer lock over the shared GPU.
--
--  The LLM engine (this process) and the image daemon (aspida-imgd) each run
--  their own CUDA context on the SAME physical GPU. Doing heavy CUDA work in
--  both at once corrupts the LLM's context (CUDA illegal-memory-access crash) —
--  process + symbol isolation does not extend to the hardware. So the two
--  serialise here: every LLM generation holds a SHARED lock for its whole
--  duration (prefill + decode); the image daemon takes the EXCLUSIVE lock
--  around a generation. Shared holders run concurrently (LLM batching is
--  preserved); the exclusive image lock waits for them to drain and blocks new
--  ones, so LLM and image GPU work never overlap.
--
--  Backed by flock(2) on /tmp/aspida_gpu.lock. Never raises: a lock failure
--  degrades to "no serialisation" rather than dropping the request.
------------------------------------------------------------------------------

package GPU_Lock is

   type Handle is limited private;

   --  Take the shared lock (blocks only while the image daemon holds exclusive).
   procedure Acquire_Shared (H : out Handle);

   --  Release + close. Idempotent; safe on an unacquired handle.
   procedure Release (H : in out Handle);

private
   type Handle is record
      Fd : Integer := -1;
   end record;
end GPU_Lock;
