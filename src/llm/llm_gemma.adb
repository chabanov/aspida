---------------------------------------------------------------------
-- LLM_Gemma body — gemma4 (Gemma-4-E4B) loader + forward.
--
-- The forward graph follows llama.cpp's models/gemma4.cpp exactly: scaled
-- token + per-layer (PLE) embeddings, QK-norm attention with V-norm and a
-- 1.0 scale, dual RoPE (full-attn layers add proportional rope_freqs),
-- per-layer head_dim (256 SWA / 512 full), shared-KV reuse for the trailing
-- layers, GeGLU FFN with sandwich norms, tied output + final logit soft-cap.
-- Decoding keeps an incremental K/V cache, so each step forwards only the new
-- token (O(n) per token rather than recomputing the whole prefix).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Unchecked_Deallocation;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;
with LLM_Tokenizer;
with LLM_RMSNorm;
with LLM_RoPE;
with LLM_Weight;
with LLM_Step_Lock;

package body LLM_Gemma is

   use Ada.Strings.Fixed;
   use type LLM_Qwen.Role_Kind;

   function Img (N : Integer) return String is
     (Trim (Integer'Image (N), Ada.Strings.Both));

   Dbg : constant Boolean := Ada.Environment_Variables.Exists ("ASPIDA_DBG");

   --  Debug: print a label plus the RMS and max-abs of a tensor's elements.
   procedure Dump (Label : String; T : Tensor) is
      N  : constant Integer := Numel (T);
      SS : Float := 0.0;
      MX : Float := 0.0;
   begin
      if not Dbg then return; end if;
      for I in 1 .. N loop
         declare V : constant Float := Get_Flat (T, I); begin
            SS := SS + V * V;
            if abs V > MX then MX := abs V; end if;
         end;
      end loop;
      Ada.Text_IO.Put_Line (Label & " n=" & Img (N)
        & " rms=" & Float'Image (Sqrt (SS / Float (N)))
        & " max=" & Float'Image (MX));
   end Dump;

   --  K/V cache: one growing column of per-head K/V vectors per attention
   --  layer (only own-KV layers are populated; shared layers point at one).
   type Tensor_Array is array (Positive range <>) of Tensor;
   type Tensor_Array_Ptr is access Tensor_Array;
   type KV_Layer is record
      K, V : Tensor_Array_Ptr;   -- each entry [1, N_KV*Head_Dim]
   end record;
   type KV_Cache is array (Positive range <>) of KV_Layer;

   type G_Block is record
      Attn_Norm, Post_Attn_Norm     : Tensor;   -- raw weights (no +1)
      Ffn_Norm, Post_Ffw_Norm       : Tensor;
      Post_Norm                     : Tensor;
      Q_Norm, K_Norm                : Tensor;
      W_Q, W_K, W_V, W_O            : LLM_Weight.Weight;
      W_Gate, W_Up, W_Down          : LLM_Weight.Weight;
      Inp_Gate, Proj                : LLM_Weight.Weight;
      Layer_Out_Scale               : Float := 1.0;
      Is_SWA                        : Boolean := True;
      Head_Dim                      : Integer := 0;     -- 256 (SWA) / 512 (full)
      N_KV                          : Integer := 0;     -- per-block KV head count
      No_V                          : Boolean := False; -- no attn_v => V := K
      Has_KV                        : Boolean := True;  -- false => reuse cache
   end record;

   type Block_Arr is array (Positive range <>) of G_Block;
   type Block_Arr_Ptr is access Block_Arr;

   type Gemma_Model_Rec is record
      Tok_Emb       : LLM_Weight.Weight;   -- token_embd (lookup + tied output)
      --  per_layer_token_embd is >2 GiB (won't fit one Ada String): keep the
      --  GGUF open and stream the row for each token on demand.
      Gf            : GGUF_File;
      PLE_Tok_Info  : Tensor_Info;
      PLE_Row_Bytes : Natural := 0;
      PLE_Proj      : LLM_Weight.Weight;   -- per_layer_model_proj
      PLE_Proj_Norm : Tensor;              -- per_layer_proj_norm
      Has_PLE       : Boolean := True;     -- Edge gemma4 (E2B/E4B); 12B/26B = False
      Out_Norm      : Tensor;
      Rope_Freqs    : Tensor;              -- proportional-RoPE divisors (full)
      Blocks        : Block_Arr_Ptr;
      Dim, N_Blocks, N_Heads, N_KV, Head_Dim, FFN, Vocab, Ctx : Integer := 0;
      HD_SWA, HD_Full : Integer := 0;      -- per-type head dims (256 / 512)
      N_KV_From_Start : Integer := 0;      -- layers [0..N-1] own their KV
      Logit_Softcap   : Float := 30.0;
      PL_Dim         : Integer := 256;     -- per-layer input dim
      Sliding_Window : Integer := 512;
      RoPE_Glob, RoPE_SWA : LLM_RoPE.RoPE_Params;
      Tok       : LLM_Tokenizer.Tokenizer;
      Bos, Eos, SOT, EOT : Integer := -1;
      --  Harmony chat format (gemma-4-*-it): channel + thinking markers.
      --  Chan_Open/Close bracket a reasoning channel; Is_Harmony gates the
      --  generation-prompt "<|channel>thought\n<channel|>" prefill and the
      --  stripping of channel content from the visible output.
      Chan_Open, Chan_Close, Think_Tok : Integer := -1;
      Is_Harmony : Boolean := False;
      --  Whether the model's own chat_template prefills an empty thought
      --  channel after "<|turn>model\n" (the non-thinking default). The 12B/26B
      --  templates do; the E4B template does not (E4B answers straight away).
      Use_Chan_Prefill : Boolean := False;
   end record;

   --  rmsnorm(x) * weight (Gemma's +1 is already folded into the GGUF weights;
   --  the eval-callback graph multiplies by the weight directly).
   function GN (X, W : Tensor) return Tensor is (LLM_RMSNorm.Forward (X, W));

   --  Stream + dequantize one row of the (>2 GiB) per-layer embedding table.
   function PLE_Row (M : Gemma_Model; Tok : Integer) return Tensor is
      RI : Tensor_Info := M.PLE_Tok_Info;
      B  : aliased String (1 .. M.PLE_Row_Bytes);
   begin
      RI.N_Dims := 2;
      RI.Dims   := [M.PLE_Tok_Info.Dims (1), 1, 0, 0];
      Read_Tensor_Range (M.Gf, M.PLE_Tok_Info,
        U64 (Tok) * U64 (M.PLE_Row_Bytes), B'Address, M.PLE_Row_Bytes);
      return LLM_Dequant.Dequantize (RI, B);
   end PLE_Row;

   --  Slice [Lo .. Lo+Len-1] (1-based, flat) of T into a fresh [1, Len] tensor.
   function Slice (T : Tensor; Lo, Len : Integer) return Tensor is
      R : Tensor := New_Tensor ([1, Len]);
   begin
      for I in 1 .. Len loop Set_Flat (R, I, Get_Flat (T, Lo + I - 1)); end loop;
      return R;
   end Slice;

   --------------------------------------------------------------------
   -- Load
   --------------------------------------------------------------------

   function Load (Path : String) return Gemma_Model is
      M : constant Gemma_Model := new Gemma_Model_Rec;
      G : GGUF_File renames M.Gf;   -- opened in place; kept open after Load

      function MI (Key : String; D : Integer) return Integer is
         V : constant String := Metadata (G, "gemma4." & Key);
      begin
         return (if V = "" then D else Integer'Value (V));
      exception when others => return D; end MI;

      function MF (Key : String; D : Float) return Float is
         V : constant String := Metadata (G, "gemma4." & Key);
      begin
         return (if V = "" then D else Float'Value (V));
      exception when others => return D; end MF;

      function LQ (Name : String) return LLM_Weight.Weight is
         Info : constant Tensor_Info := Find_Tensor (G, Name);
         Size : constant Natural := Natural (Tensor_Byte_Size (Info));
         B    : constant LLM_Weight.Byte_Data := new String (1 .. Size);
      begin
         Read_Tensor_Raw (G, Info, B.all'Address, Size);
         return LLM_Weight.From_Quant (Info, B);
      exception
         when E : others =>
            raise Model_Load_Error with "weight " & Name & ": "
              & Ada.Exceptions.Exception_Message (E);
      end LQ;

      --  A 1-D norm/scale tensor as F32 (the whole tensor is "row 0").
      function LT (Name : String) return Tensor is (LLM_Weight.Get_Row (LQ (Name), 0));

      function Has (Name : String) return Boolean is
      begin
         declare U : constant Tensor_Info := Find_Tensor (G, Name); begin
            pragma Unreferenced (U); return True;
         end;
      exception when others => return False; end Has;

      Glob_Base, SWA_Base : Float;
   begin
      Ada.Text_IO.Put_Line ("Loading Gemma (gemma4) model from " & Path & " ...");
      Open (G, Path);
      if not Is_Open (G) then
         raise Model_Load_Error with "cannot open GGUF file: " & Path;
      end if;

      M.Dim      := MI ("embedding_length", 2560);
      M.N_Blocks := MI ("block_count", 42);
      M.N_Heads  := MI ("attention.head_count", 8);
      M.N_KV     := MI ("attention.head_count_kv", 2);
      M.FFN      := MI ("feed_forward_length", 10240);
      M.Ctx      := MI ("context_length", 131072);
      M.PL_Dim   := MI ("embedding_length_per_layer_input", 256);
      M.Sliding_Window := MI ("attention.sliding_window", 512);
      M.HD_Full  := MI ("attention.key_length", 512);
      M.HD_SWA   := MI ("attention.key_length_swa", 256);
      M.Logit_Softcap := MF ("final_logit_softcapping", 30.0);
      --  Last `shared_kv_layers` blocks reuse an earlier block's K/V cache.
      M.N_KV_From_Start := M.N_Blocks - MI ("attention.shared_kv_layers", 0);
      Glob_Base := MF ("rope.freq_base", 1_000_000.0);
      SWA_Base  := MF ("rope.freq_base_swa", 10_000.0);

      M.Tok_Emb       := LQ ("token_embd.weight");
      --  Per-layer embeddings exist only on the Edge gemma4 (E2B/E4B); the
      --  larger 12B/26B variants omit them. Load PLE tensors only if present.
      M.Has_PLE := Has ("per_layer_token_embd.weight");
      if M.Has_PLE then
         declare
            I  : constant Tensor_Info := Find_Tensor (G, "per_layer_token_embd.weight");
            RI : Tensor_Info := I;
         begin
            M.PLE_Tok_Info := I;
            RI.N_Dims := 2; RI.Dims := [I.Dims (1), 1, 0, 0];
            M.PLE_Row_Bytes := Natural (Tensor_Byte_Size (RI));
         end;
         M.PLE_Proj      := LQ ("per_layer_model_proj.weight");
         M.PLE_Proj_Norm := LT ("per_layer_proj_norm.weight");
      end if;
      M.Out_Norm      := LT ("output_norm.weight");
      M.Rope_Freqs    := LT ("rope_freqs.weight");
      M.Vocab         := LLM_Weight.Rows (M.Tok_Emb);
      M.Blocks        := new Block_Arr (1 .. M.N_Blocks);

      for I in 1 .. M.N_Blocks loop
         declare
            P  : constant String := "blk." & Img (I - 1) & ".";
            Bk : G_Block;
         begin
            Bk.Is_SWA  := (I - 1) mod 6 /= 5;            -- 5 local : 1 global
            Bk.Has_KV  := (I - 1) < M.N_KV_From_Start;   -- else reuse cache
            Bk.Attn_Norm      := LT (P & "attn_norm.weight");
            Bk.Post_Attn_Norm := LT (P & "post_attention_norm.weight");
            Bk.Ffn_Norm       := LT (P & "ffn_norm.weight");
            Bk.Post_Ffw_Norm  := LT (P & "post_ffw_norm.weight");
            if M.Has_PLE then
               Bk.Post_Norm := LT (P & "post_norm.weight");
            end if;
            Bk.Q_Norm         := LT (P & "attn_q_norm.weight");
            Bk.W_Q := LQ (P & "attn_q.weight");
            Bk.W_O := LQ (P & "attn_output.weight");
            --  Shared-KV layers (the last `shared_kv_layers`) do not own K/V;
            --  they reuse an earlier block's cache, so skip those tensors.
            if Bk.Has_KV then
               Bk.K_Norm := LT (P & "attn_k_norm.weight");
               Bk.W_K := LQ (P & "attn_k.weight");
               --  gemma4 "alternative attention": attn_v is optional; when a
               --  layer has no V projection (the 12B/26B global/MQA layers),
               --  V reuses the K projection (Vcur = Kcur).
               Bk.No_V := not Has (P & "attn_v.weight");
               if not Bk.No_V then
                  Bk.W_V := LQ (P & "attn_v.weight");
               end if;
            end if;
            Bk.W_Gate := LQ (P & "ffn_gate.weight");
            Bk.W_Up   := LQ (P & "ffn_up.weight");
            Bk.W_Down := LQ (P & "ffn_down.weight");
            if M.Has_PLE then
               Bk.Inp_Gate := LQ (P & "inp_gate.weight");
               Bk.Proj     := LQ (P & "proj.weight");
            end if;
            Bk.Layer_Out_Scale :=
              Get_Flat (LLM_Weight.Get_Row (LQ (P & "layer_output_scale.weight"), 0), 1);
            Bk.Head_Dim := LLM_Weight.Rows (Bk.W_Q) / M.N_Heads;
            --  Per-block KV head count (SWA & global layers can differ — e.g.
            --  12B global layers are MQA with one KV head).
            Bk.N_KV := (if Bk.Has_KV
                        then LLM_Weight.Rows (Bk.W_K) / Bk.Head_Dim
                        else M.N_KV);
            M.Blocks (I) := Bk;
         end;
      end loop;

      M.Head_Dim := M.Blocks (1).Head_Dim;
      --  Full-attention layers: head_dim 512, base 1e6, proportional RoPE.
      M.RoPE_Glob := LLM_RoPE.Create_Qwen_RoPE (M.HD_Full, Glob_Base, M.Ctx);
      LLM_RoPE.Set_Freq_Factors (M.RoPE_Glob, M.Rope_Freqs);
      --  Sliding-window layers: head_dim 256, base 1e4, plain RoPE.
      M.RoPE_SWA  := LLM_RoPE.Create_Qwen_RoPE (M.HD_SWA, SWA_Base, M.Ctx);

      M.Tok := LLM_Tokenizer.Create;
      LLM_Tokenizer.Load_From_GGUF (M.Tok, G);
      begin M.Bos := Integer'Value (Metadata (G, "tokenizer.ggml.bos_token_id"));
      exception when others => M.Bos := 2; end;
      begin M.Eos := Integer'Value (Metadata (G, "tokenizer.ggml.eos_token_id"));
      exception when others => M.Eos := 1; end;
      --  Turn markers. Stock gemma4 uses <start_of_turn>/<end_of_turn>; some
      --  finetunes (e.g. the E4B HauhauCS one) rename them <|turn>/<turn|>.
      --  Prefer the stock pair, fall back to the finetune pair.
      M.SOT := LLM_Tokenizer.Token_To_Id (M.Tok, "<start_of_turn>");
      M.EOT := LLM_Tokenizer.Token_To_Id (M.Tok, "<end_of_turn>");
      if M.SOT < 0 or else M.EOT < 0 then
         M.SOT := LLM_Tokenizer.Token_To_Id (M.Tok, "<|turn>");
         M.EOT := LLM_Tokenizer.Token_To_Id (M.Tok, "<turn|>");
      end if;
      --  Harmony format markers (present only on the stock gemma-4 *-it models).
      M.Chan_Open  := LLM_Tokenizer.Token_To_Id (M.Tok, "<|channel>");
      M.Chan_Close := LLM_Tokenizer.Token_To_Id (M.Tok, "<channel|>");
      M.Think_Tok  := LLM_Tokenizer.Token_To_Id (M.Tok, "<|think|>");
      M.Is_Harmony := M.Chan_Open >= 0 and then M.Chan_Close >= 0;
      --  Only prefill the empty thought channel if the model's own template
      --  does (the literal "<|channel>thought\n<channel|>" — backslash-n — in
      --  its add_generation_prompt block). 12B/26B: yes; E4B: no.
      M.Use_Chan_Prefill := M.Is_Harmony and then
        Index (Metadata (G, "tokenizer.chat_template"),
               "<|channel>thought\n<channel|>") > 0;
      --  Stop generation on <end_of_turn> as well as the model's EOS.

      Ada.Text_IO.Put_Line ("  gemma4: dim=" & Img (M.Dim)
        & " layers=" & Img (M.N_Blocks) & " heads=" & Img (M.N_Heads)
        & "/" & Img (M.N_KV) & " head_dim=" & Img (M.HD_SWA) & "/" & Img (M.HD_Full)
        & " pl=" & Img (M.PL_Dim) & " kv_from_start=" & Img (M.N_KV_From_Start)
        & " vocab=" & Img (M.Vocab)
        & " bos/eos=" & Img (M.Bos) & "/" & Img (M.Eos)
        & " sot/eot=" & Img (M.SOT) & "/" & Img (M.EOT)
        & " harmony=" & (if M.Is_Harmony then "yes" else "no")
        & " chan=" & Img (M.Chan_Open) & "/" & Img (M.Chan_Close)
        & " prefill=" & (if M.Use_Chan_Prefill then "yes" else "no"));
      return M;   -- G (= M.Gf) stays open for on-demand PLE row reads
   end Load;

   --------------------------------------------------------------------
   -- One incremental decode step: forward the single token Tok at 0-based
   -- position Pos, append its K/V to the cache, and return the logits.
   -- The K/V of every prior position are read from Cache, so a step is O(seq)
   -- rather than recomputing the whole prefix.  (Cache is `in`: the access
   -- columns are fixed, but their designated elements are written here.)
   --------------------------------------------------------------------

   function Forward_Step
     (M : Gemma_Model; Cache : KV_Cache; Tok : Integer; Pos : Integer)
      return Tensor
   is
      D    : constant Integer := M.Dim;
      NH   : constant Integer := M.N_Heads;
      PL   : constant Integer := M.PL_Dim;
      --  Gemma 4 uses self.scaling = 1.0 (NO 1/sqrt(head_dim) pre-attn scale).
      AScale  : constant Float := 1.0;
      E_Scale : constant Float := Sqrt (Float (D));
      P_Scale : constant Float := Sqrt (Float (PL));
      Inv_D   : constant Float := 1.0 / Sqrt (Float (D));
      Inv_2   : constant Float := 1.0 / Sqrt (2.0);

      procedure Add_To (A : in out Tensor; B : Tensor) is
      begin
         for I in 1 .. D loop Set_Flat (A, I, Get_Flat (A, I) + Get_Flat (B, I)); end loop;
      end Add_To;

      --  RMS-normalize a single head's vector WITHOUT a weight (Gemma's V-norm).
      function VNorm (X : Tensor) return Tensor is
         N  : constant Integer := Numel (X);
         SS : Float := 0.0;
         R  : Tensor := New_Tensor ([1, N]);
      begin
         for I in 1 .. N loop SS := SS + Get_Flat (X, I) ** 2; end loop;
         declare
            Inv : constant Float := 1.0 / Sqrt (SS / Float (N) + 1.0e-6);
         begin
            for I in 1 .. N loop Set_Flat (R, I, Get_Flat (X, I) * Inv); end loop;
         end;
         return R;
      end VNorm;

      H   : Tensor := LLM_Weight.Get_Row (M.Tok_Emb, Tok);   -- residual [1, D]
      --  per-layer inputs (only used when Has_PLE; dummy [1,1] for non-PLE 12B/26B)
      IPL : Tensor := New_Tensor ([1, Integer'Max (1, M.N_Blocks * PL)]);
   begin
      --  Scaled token embedding + per-layer (PLE) inputs for this token.
      for I in 1 .. D loop Set_Flat (H, I, Get_Flat (H, I) * E_Scale); end loop;
      if M.Has_PLE then
         declare
            PTok  : constant Tensor := PLE_Row (M, Tok);
            --  per_layer_model_proj is applied to the SCALED embedding.
            PProj : Tensor := LLM_Weight.MatVec (M.PLE_Proj, H);
         begin
            for I in 1 .. M.N_Blocks * PL loop
               Set_Flat (PProj, I, Get_Flat (PProj, I) * Inv_D);
            end loop;
            for Lr in 0 .. M.N_Blocks - 1 loop
               declare
                  Sel : constant Tensor := Slice (PTok, Lr * PL + 1, PL);
                  Prj : constant Tensor :=
                    GN (Slice (PProj, Lr * PL + 1, PL), M.PLE_Proj_Norm);
               begin
                  for I in 1 .. PL loop
                     Set_Flat (IPL, Lr * PL + I,
                       (Get_Flat (Prj, I) + Get_Flat (Sel, I) * P_Scale) * Inv_2);
                  end loop;
               end;
            end loop;
         end;
      end if;

      for Lr in 1 .. M.N_Blocks loop
         declare
            B    : G_Block renames M.Blocks (Lr);
            HD   : constant Integer := B.Head_Dim;   -- 256 (SWA) / 512 (full)
            RoPE : constant LLM_RoPE.RoPE_Params :=
              (if B.Is_SWA then M.RoPE_SWA else M.RoPE_Glob);
            Win  : constant Integer :=
              (if B.Is_SWA then M.Sliding_Window else Pos + 1);
            --  Shared-KV layers attend the last own-KV layer of their type:
            --  SWA -> N_KV_From_Start-1, full -> N_KV_From_Start (1-based).
            Src  : constant Integer :=
              (if B.Has_KV then Lr
               elsif B.Is_SWA then M.N_KV_From_Start - 1
               else M.N_KV_From_Start);
            X    : constant Tensor := GN (H, B.Attn_Norm);
            Q    : Tensor := LLM_Weight.MatVec (B.W_Q, X);
         begin
            --  Q: per-head RMS-norm + RoPE at this position.
            for Hh in 0 .. NH - 1 loop
               declare
                  S : constant Tensor := LLM_RoPE.Apply
                    (RoPE, GN (Slice (Q, Hh * HD + 1, HD), B.Q_Norm), Pos);
               begin
                  for J in 1 .. HD loop Set_Flat (Q, Hh * HD + J, Get_Flat (S, J)); end loop;
               end;
            end loop;

            --  Own-KV layers compute K (norm+RoPE) and V (norm) for this token
            --  and append them to the cache column at this position. A layer
            --  without a V projection reuses K (gemma4 alternative attention);
            --  N_KV is per-block (global MQA layers have a single KV head).
            if B.Has_KV then
               declare
                  BKV : constant Integer := B.N_KV;
                  K : Tensor := LLM_Weight.MatVec (B.W_K, X);
                  V : Tensor := (if B.No_V then K else LLM_Weight.MatVec (B.W_V, X));
               begin
                  for Hh in 0 .. BKV - 1 loop
                     declare
                        Kn : constant Tensor := LLM_RoPE.Apply
                          (RoPE, GN (Slice (K, Hh * HD + 1, HD), B.K_Norm), Pos);
                        Vn : constant Tensor := VNorm (Slice (V, Hh * HD + 1, HD));
                     begin
                        for J in 1 .. HD loop
                           Set_Flat (K, Hh * HD + J, Get_Flat (Kn, J));
                           Set_Flat (V, Hh * HD + J, Get_Flat (Vn, J));
                        end loop;
                     end;
                  end loop;
                  Cache (Lr).K (Pos + 1) := K;
                  Cache (Lr).V (Pos + 1) := V;
               end;
            end if;

            --  Causal (+ sliding-window) attention over cached positions; 1.0.
            declare
               KC    : Tensor_Array_Ptr renames Cache (Src).K;
               VC    : Tensor_Array_Ptr renames Cache (Src).V;
               Lo    : constant Integer := Integer'Max (0, Pos - Win + 1);
               Ctx_O : Tensor := New_Tensor ([1, NH * HD]);
            begin
               for Hh in 0 .. NH - 1 loop
                  declare
                     KV  : constant Integer := Hh / (NH / B.N_KV);
                     Scr : Tensor := New_Tensor ([1, Pos + 1]);
                     Mx  : Float := Float'First;
                     Den : Float := 0.0;
                  begin
                     for S in Lo .. Pos loop
                        declare Dp : Float := 0.0; begin
                           for J in 1 .. HD loop
                              Dp := Dp + Get_Flat (Q, Hh * HD + J)
                                       * Get_Flat (KC (S + 1), KV * HD + J);
                           end loop;
                           Set_Flat (Scr, S + 1, Dp * AScale);
                           Mx := Float'Max (Mx, Dp * AScale);
                        end;
                     end loop;
                     for S in Lo .. Pos loop
                        Set_Flat (Scr, S + 1, Exp (Get_Flat (Scr, S + 1) - Mx));
                        Den := Den + Get_Flat (Scr, S + 1);
                     end loop;
                     for J in 1 .. HD loop
                        declare Acc : Float := 0.0; begin
                           for S in Lo .. Pos loop
                              Acc := Acc + (Get_Flat (Scr, S + 1) / Den)
                                       * Get_Flat (VC (S + 1), KV * HD + J);
                           end loop;
                           Set_Flat (Ctx_O, Hh * HD + J, Acc);
                        end;
                     end loop;
                  end;
               end loop;

               --  Post-attn norm + residual, GeGLU FFN with sandwich norms,
               --  per-layer-embedding injection, and the layer output scale.
               declare
                  Attn_Out : Tensor :=
                    GN (LLM_Weight.MatVec (B.W_O, Ctx_O), B.Post_Attn_Norm);
               begin
                  Add_To (Attn_Out, H);                          -- attn residual
                  declare
                     Xf   : constant Tensor := GN (Attn_Out, B.Ffn_Norm);
                     Gate : constant Tensor := Gelu (LLM_Weight.MatVec (B.W_Gate, Xf));
                     Up   : constant Tensor := LLM_Weight.MatVec (B.W_Up, Xf);
                     Ff   : Tensor := GN (LLM_Weight.MatVec (B.W_Down, Gate * Up),
                                          B.Post_Ffw_Norm);
                  begin
                     Add_To (Ff, Attn_Out);                      -- pe_in
                     if M.Has_PLE then
                        declare
                           Gp : constant Tensor :=
                             Gelu (LLM_Weight.MatVec (B.Inp_Gate, Ff));   -- [PL]
                           Pg : Tensor := New_Tensor ([1, PL]);
                        begin
                           for I in 1 .. PL loop
                              Set_Flat (Pg, I, Get_Flat (Gp, I)
                                * Get_Flat (IPL, (Lr - 1) * PL + I));
                           end loop;
                           declare
                              Pe : Tensor :=
                                GN (LLM_Weight.MatVec (B.Proj, Pg), B.Post_Norm);
                           begin
                              Add_To (Pe, Ff);                   -- per-layer resid
                              for I in 1 .. D loop
                                 Set_Flat (Pe, I, Get_Flat (Pe, I) * B.Layer_Out_Scale);
                              end loop;
                              H := Pe;
                           end;
                        end;
                     else
                        --  Non-PLE gemma4 (12B/26B): no per-layer embedding; the
                        --  layer output is the FFN residual times the out scale.
                        for I in 1 .. D loop
                           Set_Flat (Ff, I, Get_Flat (Ff, I) * B.Layer_Out_Scale);
                        end loop;
                        H := Ff;
                     end if;
                  end;
               end;
            end;
            if Dbg then
               Ada.Text_IO.Put_Line ("  l_out-" & Img (Lr - 1)
                 & (if B.Is_SWA then " swa" else " GLOB")
                 & ": [" & Float'Image (Get_Flat (H, 1))
                 & "," & Float'Image (Get_Flat (H, 2))
                 & "," & Float'Image (Get_Flat (H, 3))
                 & " ... " & Float'Image (Get_Flat (H, D - 2))
                 & "," & Float'Image (Get_Flat (H, D - 1))
                 & "," & Float'Image (Get_Flat (H, D)) & "]");
            end if;
         end;
      end loop;

      --  Final norm, tied output projection, then the Gemma final logit
      --  soft-cap:  logits := cap * tanh (logits / cap).
      declare
         Logits : Tensor := LLM_Weight.MatVec (M.Tok_Emb, GN (H, M.Out_Norm));
         Cap    : constant Float := M.Logit_Softcap;
      begin
         Dump ("  final H", H);
         if Cap > 0.0 then
            for I in 1 .. Numel (Logits) loop
               Set_Flat (Logits, I, Cap * Tanh (Get_Flat (Logits, I) / Cap));
            end loop;
         end if;
         if Dbg then
            declare
               Am : Integer := 1; Mv : Float := Get_Flat (Logits, 1);
            begin
               for I in 2 .. Numel (Logits) loop
                  if Get_Flat (Logits, I) > Mv then Mv := Get_Flat (Logits, I); Am := I; end if;
               end loop;
               Ada.Text_IO.Put_Line ("  logits[1..3]=" & Float'Image (Get_Flat (Logits, 1))
                 & "," & Float'Image (Get_Flat (Logits, 2))
                 & "," & Float'Image (Get_Flat (Logits, 3))
                 & " argmax=" & Img (Am - 1) & " val=" & Float'Image (Mv));
            end;
         end if;
         return Logits;
      end;
   end Forward_Step;

   --------------------------------------------------------------------
   -- Decode (sampler-driven) with an incremental K/V cache.
   --------------------------------------------------------------------

   function Decode
     (M : Gemma_Model; Prompt : LLM_Tokenizer.Token_Array;
      Max_New : Integer; Sink : access LLM_Qwen.Token_Sink'Class;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String
   is
      Cap    : constant Integer := Integer'Max (1, Prompt'Length + Max_New);
      Cache  : KV_Cache (1 .. M.N_Blocks);
      Len    : Integer := 0;        -- positions consumed so far (= next Pos)
      Out_S  : Unbounded_String;
      Logits : Tensor;
      Smp    : LLM_Sampler.Sampler := LLM_Sampler.Create (Params);
      Hist   : LLM_Sampler.History (1 .. Integer'Max (1, Max_New)) :=
        (others => 0);
      N_Hist : Natural := 0;
      In_Chan : Boolean := False;   -- inside a harmony reasoning channel?

      procedure Free is
        new Ada.Unchecked_Deallocation (Tensor_Array, Tensor_Array_Ptr);

      --  One forward step under the shared step lock, released between steps
      --  (incl. on exception) so concurrent generations interleave per token.
      function Locked_Step (Tok, Pos : Integer) return Tensor is
      begin
         LLM_Step_Lock.Acquire;
         declare
            R : constant Tensor := Forward_Step (M, Cache, Tok, Pos);
         begin
            LLM_Step_Lock.Release;
            return R;
         end;
      exception
         when others =>
            LLM_Step_Lock.Release;
            raise;
      end Locked_Step;
   begin
      --  Allocate cache columns only for layers that own a K/V.
      for L in 1 .. M.N_Blocks loop
         if M.Blocks (L).Has_KV then
            Cache (L).K := new Tensor_Array (1 .. Cap);
            Cache (L).V := new Tensor_Array (1 .. Cap);
         end if;
      end loop;

      --  Prefill the prompt; Logits ends as the distribution for the last token.
      if Dbg then
         Ada.Text_IO.Put ("  prompt tokens:");
         for I in Prompt'Range loop
            Ada.Text_IO.Put (Img (Prompt (I)) & "[" &
              LLM_Tokenizer.Decode_One (M.Tok, Prompt (I)) & "]");
         end loop;
         Ada.Text_IO.New_Line;
      end if;
      for I in Prompt'Range loop
         Logits := Locked_Step (Prompt (I), Len);
         Len := Len + 1;
      end loop;

      for Step in 1 .. Max_New loop
         declare
            Win : constant Natural :=
              Integer'Min (N_Hist, Integer'Max (0, Params.Repeat_Last_N));
            Tid : constant Integer := LLM_Sampler.Next
              (Smp, Logits, Hist (N_Hist - Win + 1 .. N_Hist));
         begin
            if Dbg then
               Ada.Text_IO.Put_Line ("  step" & Img (Step) & " -> tok " & Img (Tid)
                 & " [" & LLM_Tokenizer.Decode_One (M.Tok, Tid) & "]");
            end if;
            exit when Tid = M.Eos or else Tid = M.EOT
              or else (M.SOT >= 0 and then Tid = M.SOT);
            --  Harmony: a reasoning channel (<|channel> .. <channel|>) is the
            --  model's private thinking; bracket tokens and their content are
            --  fed back to the model but never shown to the user.
            if M.Is_Harmony and then Tid = M.Chan_Open then
               In_Chan := True;
            elsif M.Is_Harmony and then Tid = M.Chan_Close then
               In_Chan := False;
            elsif not In_Chan then
               declare Piece : constant String := LLM_Tokenizer.Decode_One (M.Tok, Tid); begin
                  Append (Out_S, Piece);
                  if Sink /= null then LLM_Qwen.Emit (Sink.all, Piece); end if;
               end;
            end if;
            N_Hist := N_Hist + 1; Hist (N_Hist) := Tid;
            exit when Len >= Cap;
            Logits := Locked_Step (Tid, Len);
            Len := Len + 1;
         end;
      end loop;

      for L in 1 .. M.N_Blocks loop
         Free (Cache (L).K);
         Free (Cache (L).V);
      end loop;
      return To_String (Out_S);
   end Decode;

   function Chat
     (M : Gemma_Model; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String
   is
      use type LLM_Tokenizer.Token_Array;
      LF : constant Character := Character'Val (10);

      function One (Id : Integer) return LLM_Tokenizer.Token_Array is
        (if Id >= 0 then LLM_Tokenizer.Token_Array'(1 => Id)
         else LLM_Tokenizer.Token_Array'(2 .. 1 => 0));

      --  Turn role label. Harmony keeps a real "system" turn; the legacy
      --  gemma/E4B template has no system role, so fold it into "user".
      function Role_Name (R : LLM_Qwen.Role_Kind) return String is
        (case R is
            when LLM_Qwen.Role_Assistant => "model",
            when LLM_Qwen.Role_System =>
              (if M.Is_Harmony then "system" else "user"),
            when LLM_Qwen.Role_User => "user");

      --  One turn: <|turn>{role}\n{content}<turn|>\n
      function Msg_Ids (Msg : LLM_Qwen.Message) return LLM_Tokenizer.Token_Array is
        (One (M.SOT)
           & LLM_Tokenizer.Encode
               (M.Tok, Role_Name (Msg.Role) & LF
                  & Trim (To_String (Msg.Text), Ada.Strings.Both))
           & One (M.EOT) & LLM_Tokenizer.Encode (M.Tok, "" & LF));

      function Conv_Ids (I : Positive) return LLM_Tokenizer.Token_Array is
      begin
         if I > Conversation'Last then return LLM_Tokenizer.Token_Array'(2 .. 1 => 0); end if;
         return Msg_Ids (Conversation (I)) & Conv_Ids (I + 1);
      end Conv_Ids;

      --  Generation prompt: open the model turn. On harmony models the
      --  non-thinking path prefills an empty "thought" channel
      --  (<|channel>thought\n<channel|>) so the model answers directly
      --  instead of emitting its own reasoning block.
      function Gen_Prompt return LLM_Tokenizer.Token_Array is
         Base : constant LLM_Tokenizer.Token_Array :=
           One (M.SOT) & LLM_Tokenizer.Encode (M.Tok, "model" & LF);
      begin
         if M.Use_Chan_Prefill then
            return Base & One (M.Chan_Open)
              & LLM_Tokenizer.Encode (M.Tok, "thought" & LF) & One (M.Chan_Close);
         else
            return Base;
         end if;
      end Gen_Prompt;
   begin
      return Decode
        (M, One (M.Bos) & Conv_Ids (Conversation'First) & Gen_Prompt,
         Max_New_Tokens, Sink, Params);
   end Chat;

   function Complete
     (M : Gemma_Model; Prompt : String; Max_New_Tokens : Integer := 8)
      return String
   is
      use type LLM_Tokenizer.Token_Array;
   begin
      return Decode
        (M,
         LLM_Tokenizer.Token_Array'(1 => M.Bos)
           & LLM_Tokenizer.Encode (M.Tok, Prompt),
         Max_New_Tokens, null);
   end Complete;

   function Vocab_Size  (M : Gemma_Model) return Integer is (M.Vocab);
   function Dim         (M : Gemma_Model) return Integer is (M.Dim);
   function Block_Count (M : Gemma_Model) return Integer is (M.N_Blocks);

end LLM_Gemma;
