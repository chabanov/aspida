---------------------------------------------------------------------
-- Student — a generic, multi-layer, bias-free Llama-style student model
-- assembled from the verified Train primitives: token embedding -> Lyr x
-- (RMSNorm + causal attention + SwiGLU MLP, with residuals) -> RMSNorm ->
-- output head. Forward caches per-layer activations; Backward runs the full
-- reverse pass and accumulates AdamW-ready gradients. This is the "scale"
-- step: depth generalizes the single block proven in test_block.
--
-- Instantiate with concrete sizes:  package S is new Student (Voc=>..,Dm=>..,
-- Ff=>.., Seq=>.., Lyr=>..);
---------------------------------------------------------------------

with Train;
with GGUF_Write;

generic
   Voc   : Positive;   -- vocabulary
   Dm    : Positive;   -- model dim
   Ff    : Positive;   -- ffn dim
   Seq   : Positive;   -- sequence length
   Lyr   : Positive;   -- number of transformer blocks
   Heads : Positive := 1;   -- attention heads (head_dim = Dm/Heads)
   Use_RoPE  : Boolean    := False;     -- RoPE instead of learned positions
   Rope_Base : Long_Float := 10000.0;   -- (Llama-compatible when Use_RoPE)
   Use_QAT   : Boolean    := False;     -- quantization-aware training: forward
   QAT_Bits  : Positive   := 8;         --  uses fake-quantized weights (STE)
package Student is

   subtype Logit_Mat is Train.Matrix (1 .. Seq, 1 .. Voc);

   type Model is limited private;

   procedure Init (M : in out Model; Seed : Long_Float);

   --  Forward over a token sequence -> per-position logits; caches activations.
   procedure Forward
     (M : in out Model; Tokens : Train.Label_Array; Logits : out Logit_Mat);

   --  Reverse pass against a teacher target [Seq x Voc]; returns the KL loss
   --  and fills the gradient buffers. Call after Forward.
   function Backward (M : in out Model; Target : Logit_Mat) return Train.Real;

   --  AdamW update of every parameter.
   procedure Step (M : in out Model; LR : Train.Real := 5.0E-3;
                   Clip : Train.Real := 0.0);

   --  Persist / restore the trained parameters (weights + embeddings, NOT the
   --  optimizer state). The file carries the architecture (Voc/Dm/Ff/Seq/Lyr/
   --  Heads); Load raises Bad_Checkpoint if it does not match this instance.
   Bad_Checkpoint : exception;
   procedure Save (M : Model;        Path : String);
   procedure Load (M : in out Model; Path : String);

   --  Export as a Llama-architecture GGUF the Aspida inference engine can run.
   --  Faithful only when Use_RoPE is True (so the engine's RoPE matches what
   --  the model was trained with) and Heads divides Dm evenly. Tokens supplies
   --  the Voc vocabulary strings.
   --  Weight-matrix quantization for export: Q_None = F32, Q_Q8_0 = ~4x
   --  smaller, Q_Q4_0/Q_Q4_K = ~8x smaller (Q4_K is the higher-quality
   --  K-quant: per-32 affine under a shared super-block scale), Q_Q6_K = ~6x
   --  smaller at ~3% error (per-16 signed scale, for sensitive tensors). The
   --  K-quants need ne0 a multiple of 256. Norms always stay F32. The engine
   --  serves the quantized GGUF directly.
   type Quant_Format is (Q_None, Q_Q8_0, Q_Q4_0, Q_Q4_K, Q_Q6_K);
   procedure Export_GGUF
     (M : Model; Path : String; Tokens : GGUF_Write.Str_List;
      Bos, Eos : Natural := 0; Ctx : Natural := 256;
      Fmt : Quant_Format := Q_None);

private

   use Train;

   subtype DMat is Matrix (1 .. Seq, 1 .. Dm);
   subtype FMat is Matrix (1 .. Seq, 1 .. Ff);
   subtype AMat is Matrix (1 .. Heads * Seq, 1 .. Seq);   -- per-head stacked
   subtype Gam  is Matrix (1 .. 1,  1 .. Dm);
   subtype WDD  is Matrix (1 .. Dm, 1 .. Dm);
   subtype WDF  is Matrix (1 .. Dm, 1 .. Ff);
   subtype WFD  is Matrix (1 .. Ff, 1 .. Dm);

   type Blk is record
      --  parameters
      G1, G2 : Gam;
      Wq, Wk, Wv, Wo : WDD;
      Wg, Wu : WDF;
      Wd : WFD;
      --  gradients
      dG1, dG2 : Gam;
      dWq, dWk, dWv, dWo : WDD;
      dWg, dWu : WDF;
      dWd : WFD;
      --  AdamW state
      AG1, AG2 : Adam (1, Dm);
      AWq, AWk, AWv, AWo : Adam (Dm, Dm);
      AWg, AWu : Adam (Dm, Ff);
      AWd : Adam (Ff, Dm);
      --  cached forward activations (for the backward pass)
      Inp, Xn1, Q, K, V, Oa, H2, Xn2 : DMat;
      A : AMat;
      Gpre, Gate, Up, Hid : FMat;
   end record;

   type Blk_Arr is array (1 .. Lyr) of Blk;

   type Model is limited record
      E    : Matrix (1 .. Voc, 1 .. Dm);
      dE   : Matrix (1 .. Voc, 1 .. Dm);
      AE   : Adam (Voc, Dm);
      Gf   : Gam;  dGf : Gam;  AGf : Adam (1, Dm);
      Wout : Matrix (1 .. Dm, 1 .. Voc);
      dWout : Matrix (1 .. Dm, 1 .. Voc);
      AWout : Adam (Dm, Voc);
      B    : Blk_Arr;
      Pos  : DMat;  dPos : DMat;  APos : Adam (Seq, Dm);   -- learned positions
      Toks : Label_Array (1 .. Seq);
      Xemb, Hf, Xf : DMat;
      Logits : Logit_Mat;
      RG   : RNG;
   end record;

end Student;
