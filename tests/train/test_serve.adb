------------------------------------------------------------------------
-- test_serve — the "serve" half of teach->train->serve: emit a complete,
-- correctly-shaped llama-architecture GGUF (token_embd, per-block attn/ffn,
-- norms, output, tokenizer, hyperparams) and then LOAD it with the real
-- LLM_Llama backend and RUN A FORWARD via Generate. Proves a model file we
-- produce is accepted and executed by the same engine that runs the teacher.
--
-- Weights here are random (training is proven separately in test_distill_train;
-- wiring the trained weights into this exporter is mechanical). The point is
-- the export->load->run machinery: GGUF weights are [ne0=in, ne1=out] with
-- Rows=out, so each [in,out] weight is transposed on the way out.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with Train;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Test_Serve is
   Dim : constant := 16;
   F   : constant := 32;
   Vsz : constant := 24;
   Ctx : constant := 64;
   Path : constant String := "/tmp/aspida_student.gguf";
   Pass : Boolean := True;

   G : Train.RNG := Train.Seeded (5.0);

   --  random llama-shaped student weights (training layout: [in, out])
   E  : Train.Matrix (1 .. Vsz, 1 .. Dim);
   G1, G2, Gf : Train.Matrix (1 .. 1, 1 .. Dim);
   Wq, Wk, Wv, Wo : Train.Matrix (1 .. Dim, 1 .. Dim);
   Wg, Wu : Train.Matrix (1 .. Dim, 1 .. F);
   Wd : Train.Matrix (1 .. F, 1 .. Dim);
   Wout : Train.Matrix (1 .. Dim, 1 .. Vsz);

   --  [in,out] training weight -> GGUF [out rows x in cols] flat (transpose)
   function To_GGUF (W : Train.Matrix) return GGUF_Write.Float_Array is
      Inn : constant Positive := W'Length (1);
      Outd : constant Positive := W'Length (2);
      R   : GGUF_Write.Float_Array (1 .. Inn * Outd);
      Idx : Positive := 1;
   begin
      for O in 1 .. Outd loop
         for I in 1 .. Inn loop
            R (Idx) := Float (W (I, O));
            Idx := Idx + 1;
         end loop;
      end loop;
      return R;
   end To_GGUF;

   --  embedding [Vocab, Dim] -> GGUF row-major [v][d] (no transpose)
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

   procedure IW (W : out Train.Matrix) is begin Train.Init_Glorot (W, G); end IW;
begin
   Put_Line ("=== Aspida serve: export student -> load -> generate ===");
   IW (E); IW (Wq); IW (Wk); IW (Wv); IW (Wo); IW (Wg); IW (Wu); IW (Wd); IW (Wout);
   G1 := [others => [others => 1.0]];
   G2 := [others => [others => 1.0]];
   Gf := [others => [others => 1.0]];

   --  ---- export a complete llama GGUF ----
   declare
      B    : GGUF_Write.Builder;
      Toks : GGUF_Write.Str_List (1 .. Vsz);
   begin
      Toks (1) := To_Unbounded_String ("<unk>");
      Toks (2) := To_Unbounded_String ("<s>");
      Toks (3) := To_Unbounded_String ("</s>");
      for I in 4 .. Vsz loop
         Toks (I) := To_Unbounded_String ("t" & Integer'Image (I - 1));
      end loop;

      GGUF_Write.Meta_Str (B, "general.architecture", "llama");
      GGUF_Write.Meta_Str (B, "general.name", "aspida-student-serve");
      GGUF_Write.Meta_U32 (B, "llama.embedding_length", Dim);
      GGUF_Write.Meta_U32 (B, "llama.block_count", 1);
      GGUF_Write.Meta_U32 (B, "llama.attention.head_count", 1);
      GGUF_Write.Meta_U32 (B, "llama.attention.head_count_kv", 1);
      GGUF_Write.Meta_U32 (B, "llama.feed_forward_length", F);
      GGUF_Write.Meta_U32 (B, "llama.context_length", Ctx);
      GGUF_Write.Meta_U32 (B, "llama.rope.dimension_count", Dim);
      GGUF_Write.Meta_F32 (B, "llama.rope.freq_base", 10000.0);
      GGUF_Write.Meta_F32 (B, "llama.attention.layer_norm_rms_epsilon", 1.0E-5);
      GGUF_Write.Meta_Str (B, "tokenizer.ggml.model", "gpt2");
      GGUF_Write.Meta_U32 (B, "tokenizer.ggml.bos_token_id", 1);
      GGUF_Write.Meta_U32 (B, "tokenizer.ggml.eos_token_id", 2);
      GGUF_Write.Meta_U32 (B, "tokenizer.ggml.unknown_token_id", 0);
      GGUF_Write.Meta_Str_Array (B, "tokenizer.ggml.tokens", Toks);

      GGUF_Write.Add_Tensor_F32 (B, "token_embd.weight",      [Dim, Vsz], Emb_Flat (E));
      GGUF_Write.Add_Tensor_F32 (B, "output_norm.weight",     [Dim],      Norm_Flat (Gf));
      GGUF_Write.Add_Tensor_F32 (B, "output.weight",          [Dim, Vsz], To_GGUF (Wout));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_norm.weight", [Dim],      Norm_Flat (G1));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_norm.weight",  [Dim],      Norm_Flat (G2));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_q.weight",      [Dim, Dim], To_GGUF (Wq));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_k.weight",      [Dim, Dim], To_GGUF (Wk));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_v.weight",      [Dim, Dim], To_GGUF (Wv));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.attn_output.weight", [Dim, Dim], To_GGUF (Wo));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_gate.weight",   [Dim, F], To_GGUF (Wg));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_up.weight",     [Dim, F], To_GGUF (Wu));
      GGUF_Write.Add_Tensor_F32 (B, "blk.0.ffn_down.weight",   [F, Dim], To_GGUF (Wd));
      GGUF_Write.Save (B, Path);
      Put_Line ("  exported " & Path);
   end;

   --  ---- load with the real llama backend + run a forward ----
   declare
      M     : LLM_Llama.Llama_Model;
      Ids   : constant LLM_Tokenizer.Token_Array := [1, 5, 7, 3];
      Reply : Unbounded_String;
   begin
      M := LLM_Llama.Load (Path);
      Put_Line ("  loaded OK; running Generate (6 tokens)…");
      Reply := To_Unbounded_String
        (LLM_Llama.Generate (M, Ids, Max_New_Tokens => 6));
      Put_Line ("  generate returned, length =" & Integer'Image (Length (Reply)));
      Put_Line ("  output: " & To_String (Reply));
   exception
      when E : others =>
         Pass := False;
         Put_Line ("  LOAD/GENERATE FAILED: " & Ada.Exceptions.Exception_Message (E));
   end;

   New_Line;
   if Pass then
      Put_Line ("RESULT: PASS  (our engine loaded and ran a model we generated)");
   else
      Put_Line ("RESULT: FAIL");
   end if;
end Test_Serve;
