------------------------------------------------------------------------
-- Student_GPU_Export body — flat-buffer -> GGUF, mirroring the layout and the
-- quantization paths of Student.Export_GGUF exactly (only the source of the
-- weights differs: a flat GPU read-back instead of a CPU Student.Model).
------------------------------------------------------------------------

with LLM_Tensor;
with LLM_Quant;

package body Student_GPU_Export is

   use GGUF_Write;

   --  Parameter count in opt-register order (must match the shim's reg() order:
   --  per layer Wq,Wk,Wv,Wo (D*D each), G1,G2 (D each), Wg,Wu (D*F each),
   --  Wd (F*D); then Gf (D), Wh (D*V), E (V*D)).
   function Param_Count
     (Voc, Dim, Ff, Lyr : Positive) return Natural
   is
      Per_Layer : constant Natural :=
        4 * Dim * Dim          -- Wq Wk Wv Wo
        + 2 * Dim              -- G1 G2
        + 2 * Dim * Ff         -- Wg Wu
        + Ff * Dim;            -- Wd
   begin
      return Lyr * Per_Layer
        + Dim                  -- Gf
        + Dim * Voc            -- Wh
        + Voc * Dim;           -- E
   end Param_Count;

   procedure Export
     (Path      : String;
      Flat      : Student_GPU.F32_Array;
      Voc, Dim, Ff, Lyr, Heads : Positive;
      Tokens    : GGUF_Write.Str_List;
      Bos, Eos  : Natural := 0;
      Ctx       : Natural := 256;
      Rope_Base : Float   := 10_000.0;
      Fmt       : Quant_Format := Q_Q8_0)
   is
      HD   : constant Natural := Dim / Heads;
      Need : constant Natural := Param_Count (Voc, Dim, Ff, Lyr);
      B    : Builder;
      Off  : Natural := Flat'First;   -- running cursor into Flat

      function Img (N : Natural) return String is
         S : constant String := Natural'Image (N);
      begin return S (S'First + 1 .. S'Last); end Img;

      --  Read the next N raw floats (row-major [In,Out]) without transposing —
      --  used for the embedding (GGUF wants [Dim,Voc] with the same v-outer,
      --  d-inner order the shim stores E in).
      function Take_Raw (N : Positive) return Float_Array is
         R : Float_Array (1 .. N);
      begin
         for I in 1 .. N loop
            R (I) := Float (Flat (Off)); Off := Off + 1;
         end loop;
         return R;
      end Take_Raw;

      --  Read the next In*Out floats stored row-major [In,Out] and emit them
      --  TRANSPOSED to GGUF order (for O, for I) — exactly Student.To_GGUF.
      function Take_T (Inn, Outd : Positive) return Float_Array is
         Src : constant Float_Array := Take_Raw (Inn * Outd);
         R   : Float_Array (1 .. Inn * Outd);
         Idx : Positive := 1;
      begin
         for O in 0 .. Outd - 1 loop
            for I in 0 .. Inn - 1 loop
               R (Idx) := Src (Src'First + I * Outd + O);   -- src(i,o) -> dst(o,i)
               Idx := Idx + 1;
            end loop;
         end loop;
         return R;
      end Take_T;

      --  Weight matrix: quantized per Fmt when ne0 is block-aligned, else F32.
      --  Identical alignment policy to Student.Export_GGUF.Add_W.
      procedure Add_W (Name : String; Dims : Dims_Array; Data : Float_Array) is
         Ne0     : constant Natural := Dims (Dims'First);
         Aligned : constant Boolean :=
           (case Fmt is
               when Q_Q4_K | Q_Q5_K | Q_Q6_K => Ne0 mod 256 = 0,
               when Q_Q8_0 | Q_Q4_0 | Q_Q5_0 => Ne0 mod 32 = 0,
               when Q_None                   => False);
      begin
         if Aligned then
            declare
               T : LLM_Tensor.Tensor := LLM_Tensor.New_Tensor ([1, Data'Length]);
            begin
               for I in Data'Range loop
                  LLM_Tensor.Set_Flat (T, I - Data'First + 1, Data (I));
               end loop;
               case Fmt is
                  when Q_Q8_0 =>
                     Add_Tensor_Q8_0 (B, Name, Dims, LLM_Quant.Quantize_Q8_0 (T));
                  when Q_Q4_0 =>
                     Add_Tensor_Q4_0 (B, Name, Dims, LLM_Quant.Quantize_Q4_0 (T));
                  when Q_Q5_0 =>
                     Add_Tensor_Q5_0 (B, Name, Dims, LLM_Quant.Quantize_Q5_0 (T));
                  when Q_Q4_K =>
                     Add_Tensor_Q4_K (B, Name, Dims, LLM_Quant.Quantize_Q4_K (T));
                  when Q_Q5_K =>
                     Add_Tensor_Q5_K (B, Name, Dims, LLM_Quant.Quantize_Q5_K (T));
                  when Q_Q6_K =>
                     Add_Tensor_Q6_K (B, Name, Dims, LLM_Quant.Quantize_Q6_K (T));
                  when Q_None => null;
               end case;
            end;
         else
            Add_Tensor_F32 (B, Name, Dims, Data);
         end if;
      end Add_W;

      --  One transformer block's nine tensors, consumed in reg() order.
      procedure Add_Block (L : Positive) is
         P : constant String := "blk." & Img (L - 1) & ".";
         --  reg() order is Wq,Wk,Wv,Wo,G1,G2,Wg,Wu,Wd — but the GGUF must list
         --  norms before the attention weights to mirror the CPU exporter; read
         --  in reg() order into locals, then add in the CPU order.
         Wq : constant Float_Array := Take_T (Dim, Dim);
         Wk : constant Float_Array := Take_T (Dim, Dim);
         Wv : constant Float_Array := Take_T (Dim, Dim);
         Wo : constant Float_Array := Take_T (Dim, Dim);
         G1 : constant Float_Array := Take_Raw (Dim);
         G2 : constant Float_Array := Take_Raw (Dim);
         Wg : constant Float_Array := Take_T (Dim, Ff);
         Wu : constant Float_Array := Take_T (Dim, Ff);
         Wd : constant Float_Array := Take_T (Ff, Dim);
      begin
         Add_Tensor_F32 (B, P & "attn_norm.weight", [Dim], G1);
         Add_Tensor_F32 (B, P & "ffn_norm.weight",  [Dim], G2);
         Add_W (P & "attn_q.weight",      [Dim, Dim], Wq);
         Add_W (P & "attn_k.weight",      [Dim, Dim], Wk);
         Add_W (P & "attn_v.weight",      [Dim, Dim], Wv);
         Add_W (P & "attn_output.weight", [Dim, Dim], Wo);
         Add_W (P & "ffn_gate.weight",    [Dim, Ff],  Wg);
         Add_W (P & "ffn_up.weight",      [Dim, Ff],  Wu);
         Add_W (P & "ffn_down.weight",    [Ff, Dim],  Wd);
      end Add_Block;

   begin
      if Flat'Length /= Need then
         raise Bad_Length with
           "flat weight buffer is" & Flat'Length'Image
           & " floats, expected" & Need'Image
           & " for the requested architecture";
      end if;

      Meta_Str (B, "general.architecture", "llama");
      Meta_Str (B, "general.name", "aspida-gpu-student");
      Meta_U32 (B, "llama.embedding_length", Dim);
      Meta_U32 (B, "llama.block_count", Lyr);
      Meta_U32 (B, "llama.attention.head_count", Heads);
      Meta_U32 (B, "llama.attention.head_count_kv", Heads);
      Meta_U32 (B, "llama.feed_forward_length", Ff);
      Meta_U32 (B, "llama.context_length", Ctx);
      Meta_U32 (B, "llama.rope.dimension_count", HD);
      Meta_F32 (B, "llama.rope.freq_base", Rope_Base);
      Meta_F32 (B, "llama.attention.layer_norm_rms_epsilon", 1.0E-6);
      Meta_Str (B, "tokenizer.ggml.model", "gpt2");
      Meta_U32 (B, "tokenizer.ggml.bos_token_id", Bos);
      Meta_U32 (B, "tokenizer.ggml.eos_token_id", Eos);
      Meta_U32 (B, "tokenizer.ggml.unknown_token_id", 0);
      Meta_Str_Array (B, "tokenizer.ggml.tokens", Tokens);

      --  Per-layer blocks come first in reg() order; the embedding / final norm
      --  / output head are the LAST three registers (Gf, Wh, E). Read the blocks
      --  first so Off lands on Gf,Wh,E in the right place.
      for L in 1 .. Lyr loop
         Add_Block (L);
      end loop;

      declare
         Gf : constant Float_Array := Take_Raw (Dim);           -- final norm
         Wh : constant Float_Array := Take_T (Dim, Voc);        -- output head
         Em : constant Float_Array := Take_Raw (Voc * Dim);     -- token embed
      begin
         Add_Tensor_F32 (B, "output_norm.weight", [Dim], Gf);
         Add_W ("token_embd.weight", [Dim, Voc], Em);
         Add_W ("output.weight",     [Dim, Voc], Wh);
      end;

      GGUF_Write.Save (B, Path);
   end Export;

end Student_GPU_Export;
