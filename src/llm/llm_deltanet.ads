---------------------------------------------------------------------
-- LLM_DeltaNet — Gated DeltaNet linear-attention recurrence
--
-- The core of Qwen3-Next "gated delta-net" layers (3 of every 4 layers
-- in qwen35moe). Per head, a fixed-size state S [Dk, Dv] is carried and
-- updated by the gated delta rule (Mamba2 + delta rule):
--
--   q~, k~  = L2_normalize(q), L2_normalize(k)
--   retr[v] = sum_k (g * S[k,v]) * k~[k]      -- what the state predicts
--   delta[v]= beta * (v[v] - retr[v])         -- error correction
--   S[k,v]  = g * S[k,v] + k~[k] * delta[v]    -- gated write
--   o[v]    = (sum_k S[k,v] * q~[k]) / sqrt(Dk)
--
-- g (decay gate, in (0,1]) and beta (write rate) are scalar per head.
-- A full layer loops this over heads and the in/out projections + conv;
-- this unit is the recurrence itself.
---------------------------------------------------------------------

with LLM_Tensor;

package LLM_DeltaNet is

   -- Zero state [Dk, Dv] for one head.
   function Init_State (Dk, Dv : Integer) return LLM_Tensor.Tensor;

   -- One token step for one head. Q,K are [1,Dk]; V,O are [1,Dv].
   -- The head's state occupies rows Base+1 .. Base+Dk of S (Base=0 for a
   -- standalone [Dk,Dv] state; >0 to address one head's slice of a packed
   -- [n_heads*Dk, Dv] state in place, avoiding an unpack/repack copy).
   procedure Step
     (S    : in out LLM_Tensor.Tensor;
      Q, K : LLM_Tensor.Tensor;
      V    : LLM_Tensor.Tensor;
      G    : Float;
      Beta : Float;
      O    : out LLM_Tensor.Tensor;
      Base : Integer := 0);

end LLM_DeltaNet;
