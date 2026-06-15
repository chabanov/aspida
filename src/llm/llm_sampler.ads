---------------------------------------------------------------------
-- LLM_Sampler — market-standard token sampling, shared by every backend.
--
-- Given a logits vector it applies (in llama.cpp order): repetition penalty
-- over recent history, temperature, top-k, then top-p (nucleus), and draws a
-- token from the resulting distribution.  Temperature <= 0 means deterministic
-- greedy (argmax) — the previous behaviour, so existing callers are unchanged.
--
-- The PRNG is a self-contained xorshift64* (no third-party RNG), seeded for
-- reproducibility: same Seed + same logits + same history => same token.
---------------------------------------------------------------------

with LLM_Tensor;
with Interfaces;

package LLM_Sampler is

   type Params is record
      Temperature    : Float    := 0.0;   -- <= 0.0 => greedy argmax
      Top_K          : Integer  := 0;     -- <= 0   => no top-k cut
      Top_P          : Float    := 1.0;   -- 1.0    => no nucleus cut
      Repeat_Penalty : Float    := 1.0;   -- 1.0    => no penalty
      Repeat_Last_N  : Integer  := 64;    -- window of recent tokens penalised
      Seed           : Long_Long_Integer := 0;  -- 0 => fixed default seed
   end record;

   Greedy : constant Params := (others => <>);

   type Sampler is private;

   function Create (P : Params) return Sampler;

   --  Recent generated token ids (0-based), most-recent order irrelevant; only
   --  membership matters for the penalty.
   type History is array (Positive range <>) of Integer;
   Empty_History : constant History := [1 .. 0 => 0];

   --  Choose the next token id (0-based) from logits [1, vocab].
   function Next
     (S      : in out Sampler;
      Logits : LLM_Tensor.Tensor;
      Recent : History := Empty_History) return Integer;

private

   type Sampler is record
      P     : Params;
      State : Interfaces.Unsigned_64 := 0;
   end record;

end LLM_Sampler;
