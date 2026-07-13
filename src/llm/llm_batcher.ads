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

   --  Called once after the resident chain is registered (model load).
   procedure Configure (N_Layers, Vocab : Integer);

   --  Claim a batch lane for the lifetime of one generation. Lane = -1 if the
   --  pool is full (caller should fall back to the single-request path).
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
   procedure Step (Lane, Embed_Row, Pos, N_Layers : Integer;
                   Handles, Logits : System.Address);

end LLM_Batcher;
