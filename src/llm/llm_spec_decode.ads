---------------------------------------------------------------------
-- LLM_Spec_Decode — speculative decoding (draft/target), CPU reference.
--
-- A small DRAFT model proposes Gamma tokens autoregressively; the large
-- TARGET model then verifies all of them in a SINGLE forward pass. Every
-- draft token the target agrees with is accepted for free, so k accepted
-- tokens cost one target forward instead of k. Draft and target MUST share
-- a vocabulary (hura 9b / 35b do — byte-identical, verified).
--
-- CORRECTNESS INVARIANT (greedy, Temperature <= 0):
--   the sequence this produces is BYTE-IDENTICAL to what the target model
--   produces greedily on its own. Speculation is exact for greedy decoding —
--   the draft only changes SPEED, never the output. test_spec_decode asserts
--   exactly this against the real hura 9b + 35b, and it is the whole point
--   of step 1: prove the algorithm before touching the GPU.
--
-- SCOPE (step 1): this is a CPU reference built on LLM_Qwen.Forward_Logits.
-- It is correct, not fast — both models run on the CPU and each verification
-- re-encodes the growing prefix, so it is O(n^2)-ish. Its job is to be the
-- bit-exact oracle the GPU path (step 2/3) is validated against, and to
-- measure acceptance rate vs Gamma on the real model pair. It is NOT wired
-- into the server.
--
-- Sampling (Temperature > 0) uses the Leviathan/Chen acceptance rule and is
-- distributionally exact but NOT byte-identical to a single greedy run, so it
-- cannot be checked by equality; it is provided but its test is a separate
-- statistical one. Step 1 verifies the greedy path.
---------------------------------------------------------------------

with LLM_Qwen;
with LLM_Tokenizer;
with LLM_Sampler;

package LLM_Spec_Decode is

   type Stats is record
      Target_Forwards : Natural := 0;   -- verification passes (the expensive op)
      Draft_Forwards  : Natural := 0;   -- draft proposals
      Proposed        : Natural := 0;   -- draft tokens offered
      Accepted        : Natural := 0;   -- draft tokens the target agreed with
      Emitted         : Natural := 0;   -- total tokens produced
   end record;
   --  Acceptance rate = Accepted / Proposed; the closer to 1, the better the
   --  draft. Speedup ~ Emitted / Target_Forwards (tokens per expensive pass).

   --  Greedy speculative decode. Draft proposes Gamma tokens per round; target
   --  verifies. Stops at Max_New_Tokens or when the target emits Stop_Id.
   --  Params.Temperature <= 0 selects the greedy path (the verified one);
   --  Temperature > 0 selects acceptance sampling with Params.Seed.
   function Generate
     (Draft, Target  : LLM_Qwen.Qwen_Model;
      Prompt_Ids     : LLM_Tokenizer.Token_Array;
      Max_New_Tokens : Positive;
      Stop_Id        : Integer;
      Gamma          : Positive := 4;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Result_Stats   : access Stats := null)
      return LLM_Tokenizer.Token_Array;

   --  PROMPT-LOOKUP speculative decode: no draft model. The "draft" is the
   --  context itself — the last Ngram tokens are matched against earlier text
   --  and up to Gamma following tokens are proposed, then verified by the
   --  target exactly as above. Byte-identical to target-alone greedy (the
   --  target verifies every token); it only wins where the output repeats
   --  material already in context (code, quoting, structured/agentic replies).
   --  Free: no second model, no extra VRAM, and it gets MORE effective as the
   --  context grows (more to match). Where nothing repeats it costs nothing
   --  extra — each round degenerates to one ordinary target step.
   function Generate_Lookup
     (Target         : LLM_Qwen.Qwen_Model;
      Prompt_Ids     : LLM_Tokenizer.Token_Array;
      Max_New_Tokens : Positive;
      Stop_Id        : Integer;
      Ngram          : Positive := 3;
      Gamma          : Positive := 8;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Result_Stats   : access Stats := null)
      return LLM_Tokenizer.Token_Array;

end LLM_Spec_Decode;
