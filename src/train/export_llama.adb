---------------------------------------------------------------------
-- Export_Llama body.
---------------------------------------------------------------------

package body Export_Llama is

   --  [in,out] training weight -> GGUF [out rows x in cols] flat (transpose)
   function To_GGUF (W : Train.Matrix) return GGUF_Write.Float_Array is
      Inn  : constant Positive := W'Length (1);
      Outd : constant Positive := W'Length (2);
      R    : GGUF_Write.Float_Array (1 .. Inn * Outd);
      Idx  : Positive := 1;
   begin
      for O in 1 .. Outd loop
         for I in 1 .. Inn loop
            R (Idx) := Float (W (I, O));
            Idx := Idx + 1;
         end loop;
      end loop;
      return R;
   end To_GGUF;

   --  embedding [Vocab,Dim] -> row-major [v][d] (no transpose)
   function Emb_Flat (Em : Train.Matrix) return GGUF_Write.Float_Array is
      R   : GGUF_Write.Float_Array (1 .. Em'Length (1) * Em'Length (2));
      Idx : Positive := 1;
   begin
      for V in 1 .. Em'Length (1) loop
         for Dd in 1 .. Em'Length (2) loop
            R (Idx) := Float (Em (V, Dd)); Idx := Idx + 1;
         end loop;
      end loop;
      return R;
   end Emb_Flat;

   function Norm_Flat (Ga : Train.Matrix) return GGUF_Write.Float_Array is
      R : GGUF_Write.Float_Array (1 .. Ga'Length (2));
   begin
      for J in 1 .. Ga'Length (2) loop R (J) := Float (Ga (1, J)); end loop;
      return R;
   end Norm_Flat;

   procedure Save
     (Path : String;
      E, G1, G2, Gf, Wq, Wk, Wv, Wo, Wg, Wu, Wd, Wout : Train.Matrix;
      Tokens    : GGUF_Write.Str_List;
      Bos, Eos  : Natural := 1;
      Ctx       : Natural := 64;
      Rope_Base : Float   := 10000.0;
      RMS_Eps   : Float   := 1.0E-5)
   is
      Dim : constant Natural := E'Length (2);
      Vsz : constant Natural := E'Length (1);
      F   : constant Natural := Wg'Length (2);
      B   : GGUF_Write.Builder;
   begin
      GGUF_Write.Meta_Str (B, "general.architecture", "llama");
      GGUF_Write.Meta_Str (B, "general.name", "aspida-student");
      GGUF_Write.Meta_U32 (B, "llama.embedding_length", Dim);
      GGUF_Write.Meta_U32 (B, "llama.block_count", 1);
      GGUF_Write.Meta_U32 (B, "llama.attention.head_count", 1);
      GGUF_Write.Meta_U32 (B, "llama.attention.head_count_kv", 1);
      GGUF_Write.Meta_U32 (B, "llama.feed_forward_length", F);
      GGUF_Write.Meta_U32 (B, "llama.context_length", Ctx);
      GGUF_Write.Meta_U32 (B, "llama.rope.dimension_count", Dim);
      GGUF_Write.Meta_F32 (B, "llama.rope.freq_base", Rope_Base);
      GGUF_Write.Meta_F32 (B, "llama.attention.layer_norm_rms_epsilon", RMS_Eps);
      GGUF_Write.Meta_Str (B, "tokenizer.ggml.model", "gpt2");
      GGUF_Write.Meta_U32 (B, "tokenizer.ggml.bos_token_id", Bos);
      GGUF_Write.Meta_U32 (B, "tokenizer.ggml.eos_token_id", Eos);
      GGUF_Write.Meta_U32 (B, "tokenizer.ggml.unknown_token_id", 0);
      GGUF_Write.Meta_Str_Array (B, "tokenizer.ggml.tokens", Tokens);

      GGUF_Write.Add_Tensor_F32 (B, "token_embd.weight",       [Dim, Vsz], Emb_Flat (E));
      GGUF_Write.Add_Tensor_F32 (B, "output_norm.weight",      [Dim],      Norm_Flat (Gf));
      GGUF_Write.Add_Tensor_F32 (B, "output.weight",           [Dim, Vsz], To_GGUF (Wout));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_norm.weight",  [Dim],      Norm_Flat (G1));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_norm.weight",   [Dim],      Norm_Flat (G2));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_q.weight",      [Dim, Dim], To_GGUF (Wq));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_k.weight",      [Dim, Dim], To_GGUF (Wk));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_v.weight",      [Dim, Dim], To_GGUF (Wv));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_output.weight", [Dim, Dim], To_GGUF (Wo));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_gate.weight",   [Dim, F], To_GGUF (Wg));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_up.weight",     [Dim, F], To_GGUF (Wu));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_down.weight",   [F, Dim], To_GGUF (Wd));
      GGUF_Write.Save (B, Path);
   end Save;

end Export_Llama;
