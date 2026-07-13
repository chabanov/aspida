--  LLM_Batcher — continuous (dynamic) batching for the resident GPU chain.
--
--  Without this, each client's handler task serialises its per-token forward
--  through a global step lock: N clients share ONE single-request forward per
--  token, so aggregate throughput stays flat. Here the handler tasks instead
--  submit their per-token step to a shared Pool and block; a single Driver task
--  gathers all pending steps, runs ONE aspida_gpu_chain_forward_batch(B) for the
--  whole set, and hands each caller its logits. Aggregate throughput then scales
--  with the number of concurrent requests (the batched forward reads each shared
--  weight once for all lanes).
--
--  The batched forward takes no CUDA graph, so B may vary freely step to step —
--  requests join and leave the batch as clients connect and finish.
with System;
package LLM_Batcher is

   --  Enabled by env ASPIDA_BATCH_SERVE (falls back to the single-request path).
   function Enabled return Boolean;

   --  Raised by Step in every caller of a batch whose GPU forward failed (see
   --  the Driver's handler): the generation must abort — its logits were never
   --  produced. Callers unwind through Decode_Tokens (which frees GPU state)
   --  to the handler (which releases its locks and reports internal error).
   Batch_Failed : exception;

   --  Called once after the resident chain is registered (model load).
   procedure Configure (N_Layers, Vocab : Integer);

   --  Claim a batch lane for the lifetime of one generation. BLOCKS until a
   --  lane is free when the pool is full: falling back to the single-request
   --  path while the batcher is live is NOT safe (the single path drives the
   --  shared resident chain state concurrently with the Driver — the exact
   --  cross-generation KV corruption the batcher exists to avoid), so excess
   --  requests queue here instead. Lane is always >= 0 on return.
   procedure Begin_Gen (Lane : out Integer);
   procedure End_Gen (Lane : Integer);

   --  Serialised allocation of this generation's per-layer GPU state. The batch
   --  Driver is the only GPU *forward* caller, but state (Dnet_New/Fattn_New)
   --  is allocated from the handler tasks, so it must be mutually excluded.
   procedure Alloc_Lock;
   procedure Alloc_Unlock;

   --  One batched forward step for this lane. Blocks until the shared forward
   --  that includes this lane completes and its logits are in Logits.
   --  Handles -> N_Layers C ints (this request's per-layer state handles).
   --  Logits  -> Vocab floats, filled on return.
   --  Raises Batch_Failed if that forward raised (e.g. a CUDA error): the
   --  logits were not produced and this generation must abort.
   procedure Step (Lane, Embed_Row, Pos, N_Layers : Integer;
                   Handles, Logits : System.Address);

end LLM_Batcher;
