------------------------------------------------------------------------
-- Student_GPU_Export — write a servable Llama-architecture GGUF directly from
-- the FLAT trained-weight buffer read off the GPU (Student_GPU.Get_Weights).
--
-- The flat buffer is in the shim's opt-register order (per layer:
-- Wq,Wk,Wv,Wo,G1,G2,Wg,Wu,Wd; then Gf,Wh,E), every weight row-major [in,out]
-- and the norms [Dim]. This is the SAME architecture and tensor layout as the
-- CPU Student.Export_GGUF, so the GGUF this writes is bit-faithful to a CPU
-- export of the same weights — and the Aspida engine (LLM_Llama) serves it
-- directly. Quantization formats mirror Student.Quant_Format.
--
-- This is the GPU-side half of the teach->train->serve loop: the model never
-- has to be materialized as a (huge, tier-scale) CPU Student.Model to export.
------------------------------------------------------------------------

with GGUF_Write;
with Student_GPU;

package Student_GPU_Export is

   type Quant_Format is
     (Q_None, Q_Q8_0, Q_Q4_0, Q_Q5_0, Q_Q4_K, Q_Q5_K, Q_Q6_K);

   Bad_Length : exception;   -- Flat'Length /= the architecture's parameter count

   --  Write a GGUF at Path from the flat GPU weight buffer Flat.
   --  Voc/Dim/Ff/Lyr/Heads describe the trained student (Config_Of (tier));
   --  Rope_Base must match training (the GPU shim uses 10000.0). Tokens supplies
   --  the Voc vocabulary strings. Fmt selects weight quantization (norms stay
   --  F32; rows that aren't block-aligned fall back to F32).
   procedure Export
     (Path      : String;
      Flat      : Student_GPU.F32_Array;
      Voc, Dim, Ff, Lyr, Heads : Positive;
      Tokens    : GGUF_Write.Str_List;
      Bos, Eos  : Natural := 0;
      Ctx       : Natural := 256;
      Rope_Base : Float   := 10_000.0;
      Fmt       : Quant_Format := Q_Q8_0)
     with Pre => Dim mod Heads = 0 and then Tokens'Length = Voc;

   --  The expected length of Flat for a given architecture (so callers can size
   --  the read-back buffer and validate against Student_GPU.N_Params).
   function Param_Count
     (Voc, Dim, Ff, Lyr : Positive) return Natural;

end Student_GPU_Export;
