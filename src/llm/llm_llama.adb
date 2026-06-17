---------------------------------------------------------------------
-- LLM_Llama body — dense transformer (Llama 3.x / Mistral / Qwen2-dense).
--
-- Standard decoder graph: pre-attn RMSNorm, GQA attention with NeoX RoPE
-- (proportional rope_freqs when present), SwiGLU FFN, residual adds, untied
-- (or tied) output projection.  No bias, no QK-norm, no MoE, no sliding
-- window, no logit soft-cap.  Incremental K/V cache: one forward per token.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Unchecked_Deallocation;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with Ada.Environment_Variables;
with Ada.Real_Time;
with Ada.Synchronous_Task_Control;
with Ada.Unchecked_Conversion;
with System; use type System.Address;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Tensor;  use LLM_Tensor;
with LLM_RMSNorm;
with LLM_RoPE;
with LLM_Weight;
with LLM_GPU;
with LLM_Pool;
with LLM_Step_Lock;

package body LLM_Llama is

   use Ada.Strings.Fixed;

   function Img (N : Integer) return String is
     (Trim (Integer'Image (N), Ada.Strings.Both));

   type Tensor_Array is array (Positive range <>) of Tensor;
   type Tensor_Array_Ptr is access Tensor_Array;
   type KV_Layer is record
      K, V : Tensor_Array_Ptr;   -- each entry [1, N_KV*Head_Dim]
   end record;
   type KV_Cache is array (Positive range <>) of KV_Layer;

   type L_Block is record
      Attn_Norm, Ffn_Norm   : Tensor;
      W_Q, W_K, W_V, W_O     : LLM_Weight.Weight;
      W_Gate, W_Up, W_Down   : LLM_Weight.Weight;
   end record;

   type Block_Arr is array (Positive range <>) of L_Block;
   type Block_Arr_Ptr is access Block_Arr;

   type Llama_Model_Rec is record
      Tok_Emb   : LLM_Weight.Weight;   -- token_embd (lookup)
      Output    : LLM_Weight.Weight;   -- output.weight (tied to Tok_Emb if absent)
      Out_Norm  : Tensor;
      Rope_Freqs : Tensor;             -- proportional-RoPE divisors (if present)
      Has_Freqs  : Boolean := False;
      Blocks    : Block_Arr_Ptr;
      Dim, N_Blocks, N_Heads, N_KV, Head_Dim, FFN, Vocab, Ctx : Integer := 0;
      RoPE      : LLM_RoPE.RoPE_Params;
      Tok       : LLM_Tokenizer.Tokenizer;
      Bos, Eos, Eot, SH, EH : Integer := -1;
   end record;

   --  Model RMS epsilon, set from GGUF at load (Llama-3 = 1e-5). Package-level
   --  because GN has no access to the model record; the server loads one model.
   RMS_Eps : Float := 1.0e-5;
   function GN (X, W : Tensor) return Tensor is (LLM_RMSNorm.Forward (X, W, RMS_Eps));

   --  Coarse profiler (ASPIDA_PROF): split per-token wall time into the CPU
   --  attention loop vs everything else (matvecs/norms/rope/marshalling).
   Prof : constant Boolean := Ada.Environment_Variables.Exists ("ASPIDA_PROF");
   Acc_Attn, Acc_Mv, Acc_Total, Acc_Rope, Acc_Ffn : Duration := 0.0;

   Dbg : constant Boolean := Ada.Environment_Variables.Exists ("ASPIDA_DBG");
   procedure Dump (Label : String; T : Tensor) is
      N : constant Integer := Numel (T);
   begin
      if not Dbg or else N < 6 then return; end if;
      Ada.Text_IO.Put_Line (Label
        & ": [" & Float'Image (Get_Flat (T, 1)) & "," & Float'Image (Get_Flat (T, 2))
        & "," & Float'Image (Get_Flat (T, 3))
        & " ... " & Float'Image (Get_Flat (T, N - 1)) & "," & Float'Image (Get_Flat (T, N)) & "]");
   end Dump;

   --  Slice [Lo .. Lo+Len-1] (1-based, flat) of T into a fresh [1, Len] tensor.
   function Slice (T : Tensor; Lo, Len : Integer) return Tensor is
      R : Tensor := New_Tensor ([1, Len]);
   begin
      for I in 1 .. Len loop Set_Flat (R, I, Get_Flat (T, Lo + I - 1)); end loop;
      return R;
   end Slice;

   --  SwiGLU activation: silu(x) = x / (1 + exp(-x)).
   function Silu (X : Tensor) return Tensor is
      N : constant Integer := Numel (X);
      R : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop
         declare V : constant Float := Get_Flat (X, I); begin
            Set_Flat (R, I, V / (1.0 + Exp (-V)));
         end;
      end loop;
      return R;
   end Silu;

   --------------------------------------------------------------------
   -- Load
   --------------------------------------------------------------------

   function Load (Path : String) return Llama_Model is
      M : constant Llama_Model := new Llama_Model_Rec;
      G : GGUF_File;

      function MI (Key : String; D : Integer) return Integer is
         V : constant String := Metadata (G, "llama." & Key);
      begin
         return (if V = "" then D else Integer'Value (V));
      exception when others => return D; end MI;

      function MF (Key : String; D : Float) return Float is
         V : constant String := Metadata (G, "llama." & Key);
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

      function Has (Name : String) return Boolean is
      begin
         declare Unused : constant Tensor_Info := Find_Tensor (G, Name); begin
            pragma Unreferenced (Unused);
            return True;
         end;
      exception when others => return False;
      end Has;

      function LT (Name : String) return Tensor is (LLM_Weight.Get_Row (LQ (Name), 0));

      RoPE_Base : Float;
   begin
      Ada.Text_IO.Put_Line ("Loading Llama (dense) model from " & Path & " ...");
      Open (G, Path);
      if not Is_Open (G) then
         raise Model_Load_Error with "cannot open GGUF file: " & Path;
      end if;
      if Metadata (G, "general.architecture") /= "llama" then
         raise Model_Load_Error with "not a 'llama' architecture model";
      end if;

      M.Dim      := MI ("embedding_length", 4096);
      M.N_Blocks := MI ("block_count", 32);
      M.N_Heads  := MI ("attention.head_count", 32);
      M.N_KV     := MI ("attention.head_count_kv", M.N_Heads);
      M.FFN      := MI ("feed_forward_length", 11008);
      M.Ctx      := MI ("context_length", 8192);
      M.Head_Dim := MI ("rope.dimension_count", M.Dim / M.N_Heads);
      RoPE_Base  := MF ("rope.freq_base", 500_000.0);
      RMS_Eps    := MF ("attention.layer_norm_rms_epsilon", 1.0e-5);

      M.Tok_Emb  := LQ ("token_embd.weight");
      M.Out_Norm := LT ("output_norm.weight");
      --  Untied output if present, else tie to the input embedding.
      M.Output   := (if Has ("output.weight") then LQ ("output.weight") else M.Tok_Emb);
      M.Vocab    := LLM_Weight.Rows (M.Output);
      M.Has_Freqs := Has ("rope_freqs.weight");
      if M.Has_Freqs then M.Rope_Freqs := LT ("rope_freqs.weight"); end if;

      M.Blocks := new Block_Arr (1 .. M.N_Blocks);
      for I in 1 .. M.N_Blocks loop
         declare
            P  : constant String := "blk." & Img (I - 1) & ".";
            Bk : L_Block;
         begin
            Bk.Attn_Norm := LT (P & "attn_norm.weight");
            Bk.Ffn_Norm  := LT (P & "ffn_norm.weight");
            Bk.W_Q := LQ (P & "attn_q.weight");
            Bk.W_K := LQ (P & "attn_k.weight");
            Bk.W_V := LQ (P & "attn_v.weight");
            Bk.W_O := LQ (P & "attn_output.weight");
            Bk.W_Gate := LQ (P & "ffn_gate.weight");
            Bk.W_Up   := LQ (P & "ffn_up.weight");
            Bk.W_Down := LQ (P & "ffn_down.weight");
            M.Blocks (I) := Bk;
         end;
      end loop;

      M.Head_Dim := LLM_Weight.Rows (M.Blocks (1).W_Q) / M.N_Heads;
      M.RoPE := LLM_RoPE.Create_Qwen_RoPE (M.Head_Dim, RoPE_Base, M.Ctx);
      --  Llama GGUF permutes Q/K weights for interleaved (NORM) rotation.
      LLM_RoPE.Set_Interleaved (M.RoPE);
      if M.Has_Freqs and then not Ada.Environment_Variables.Exists ("ASPIDA_NO_FF") then
         LLM_RoPE.Set_Freq_Factors (M.RoPE, M.Rope_Freqs);
      end if;

      M.Tok := LLM_Tokenizer.Create;
      LLM_Tokenizer.Load_From_GGUF (M.Tok, G);
      begin M.Bos := Integer'Value (Metadata (G, "tokenizer.ggml.bos_token_id"));
      exception when others => M.Bos := LLM_Tokenizer.Token_To_Id (M.Tok, "<|begin_of_text|>"); end;
      begin M.Eos := Integer'Value (Metadata (G, "tokenizer.ggml.eos_token_id"));
      exception when others => M.Eos := LLM_Tokenizer.Token_To_Id (M.Tok, "<|end_of_text|>"); end;
      M.Eot := LLM_Tokenizer.Token_To_Id (M.Tok, "<|eot_id|>");
      M.SH  := LLM_Tokenizer.Token_To_Id (M.Tok, "<|start_header_id|>");
      M.EH  := LLM_Tokenizer.Token_To_Id (M.Tok, "<|end_header_id|>");

      Close (G);
      Ada.Text_IO.Put_Line ("  llama: dim=" & Img (M.Dim)
        & " layers=" & Img (M.N_Blocks) & " heads=" & Img (M.N_Heads)
        & "/" & Img (M.N_KV) & " head_dim=" & Img (M.Head_Dim)
        & " ffn=" & Img (M.FFN) & " vocab=" & Img (M.Vocab)
        & (if M.Has_Freqs then " rope_freqs" else ""));
      return M;
   end Load;

   --  Matvec, on the GPU when the LLM_GPU shim is loaded (Q4_K/Q5_K/Q6_K
   --  weights — those with Kind_Code >= 0), else the pure-Ada CPU path.
   --  Bit-identical kernels, so output is unchanged.
   function GMV (W : LLM_Weight.Weight; X : Tensor) return Tensor is
      use type Ada.Real_Time.Time;
      KC : constant Integer := LLM_Weight.Kind_Code (W);
      T0 : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
   begin
      if KC >= 0 and then LLM_GPU.Available then
         declare
            Ind  : constant Integer := LLM_Weight.Cols (W);
            Outd : constant Integer := LLM_Weight.Rows (W);
         begin
            --  Build the result in place (extended return) so there is NO
            --  controlled-type Adjust deep-copy when the caller assigns it.
            --  Zero-copy: pass the tensors' contiguous FP32 buffers straight to
            --  the GPU shim (Float ≡ C float here) — no per-element marshalling.
            return Y : constant Tensor := New_Tensor ([1, Outd]) do
               LLM_GPU.MatVec
                 (LLM_Weight.Raw_Address (W), LLM_Weight.Raw_Bytes (W),
                  KC, Ind, Outd, Data_Address (X), Data_Address (Y));
               if Prof then
                  Acc_Mv := Acc_Mv +
                    Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - T0);
               end if;
            end return;
         end;
      else
         return LLM_Weight.MatVec (W, X);
      end if;
   end GMV;

   --------------------------------------------------------------------
   -- One incremental decode step at 0-based position Pos.
   --------------------------------------------------------------------

   function Forward_Step
     (M : Llama_Model; Cache : KV_Cache; Tok : Integer; Pos : Integer)
      return Tensor
   is
      D   : constant Integer := M.Dim;
      NH  : constant Integer := M.N_Heads;
      NKV : constant Integer := M.N_KV;
      HD  : constant Integer := M.Head_Dim;
      AScale : constant Float := 1.0 / Sqrt (Float (HD));

      procedure Add_To (A : in out Tensor; B : Tensor) is
      begin
         for I in 1 .. D loop Set_Flat (A, I, Get_Flat (A, I) + Get_Flat (B, I)); end loop;
      end Add_To;

      H : Tensor := LLM_Weight.Get_Row (M.Tok_Emb, Tok);   -- residual [1, D]
      T_Step : Ada.Real_Time.Time;
      use type Ada.Real_Time.Time;
   begin
      T_Step := Ada.Real_Time.Clock;
      if Dbg then Dump ("  emb pos=" & Pos'Image & " tok=" & Tok'Image, H); end if;
      for Lr in 1 .. M.N_Blocks loop
         declare
            B : L_Block renames M.Blocks (Lr);
            X : constant Tensor := GN (H, B.Attn_Norm);
            Q : Tensor := GMV (B.W_Q, X);
            K : Tensor := GMV (B.W_K, X);
            V : constant Tensor := GMV (B.W_V, X);
         begin
            if Dbg and then Lr = 1 then Dump ("  attn_norm p=" & Pos'Image & " L0", X); end if;
            --  RoPE on Q (per head) and K (per kv head); no QK-norm, no bias.
            declare
               T_Rope : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
            begin
            for Hh in 0 .. NH - 1 loop
               declare
                  S : constant Tensor :=
                    LLM_RoPE.Apply (M.RoPE, Slice (Q, Hh * HD + 1, HD), Pos);
               begin
                  for J in 1 .. HD loop Set_Flat (Q, Hh * HD + J, Get_Flat (S, J)); end loop;
               end;
            end loop;
            for Hh in 0 .. NKV - 1 loop
               declare
                  S : constant Tensor :=
                    LLM_RoPE.Apply (M.RoPE, Slice (K, Hh * HD + 1, HD), Pos);
               begin
                  for J in 1 .. HD loop Set_Flat (K, Hh * HD + J, Get_Flat (S, J)); end loop;
               end;
            end loop;
            if Prof then
               Acc_Rope := Acc_Rope +
                 Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - T_Rope);
            end if;
            end;
            Cache (Lr).K (Pos + 1) := K;
            Cache (Lr).V (Pos + 1) := V;

            --  Causal GQA attention over cached positions; scale 1/sqrt(HD).
            declare
               KC    : Tensor_Array_Ptr renames Cache (Lr).K;
               VC    : Tensor_Array_Ptr renames Cache (Lr).V;
               Ctx_O : Tensor := New_Tensor ([1, NH * HD]);
               T_Attn : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;

               --  Heads are independent (each writes a disjoint Ctx_O slice and
               --  only reads Q/K/V), so fan them out across the persistent
               --  worker pool. Per-head Scr is a stack array — no allocation, so
               --  no allocator contention between worker threads.
               type Head_Op is new LLM_Pool.Parallel_Op with null record;
               overriding procedure Execute
                 (Op : in out Head_Op; Lo, Hi : Integer)
               is
               begin
                  for Hh in Lo .. Hi loop
                     declare
                        KV  : constant Integer := Hh / (NH / NKV);
                        Scr : array (0 .. Pos) of Float;
                        Mx  : Float := Float'First;
                        Den : Float := 0.0;
                     begin
                        for S in 0 .. Pos loop
                           declare Dp : Float := 0.0; begin
                              for J in 1 .. HD loop
                                 Dp := Dp + Get_Flat (Q, Hh * HD + J)
                                          * Get_Flat (KC (S + 1), KV * HD + J);
                              end loop;
                              Scr (S) := Dp * AScale;
                              Mx := Float'Max (Mx, Scr (S));
                           end;
                        end loop;
                        for S in 0 .. Pos loop
                           Scr (S) := Exp (Scr (S) - Mx);
                           Den := Den + Scr (S);
                        end loop;
                        for J in 1 .. HD loop
                           declare Acc : Float := 0.0; begin
                              for S in 0 .. Pos loop
                                 Acc := Acc + (Scr (S) / Den)
                                          * Get_Flat (VC (S + 1), KV * HD + J);
                              end loop;
                              Set_Flat (Ctx_O, Hh * HD + J, Acc);
                           end;
                        end loop;
                     end;
                  end loop;
               end Execute;

               HOp : Head_Op;
            begin
               LLM_Pool.Run (HOp, 0, NH - 1, Min_Grain => 2);
               if Prof then
                  Acc_Attn := Acc_Attn +
                    Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - T_Attn);
               end if;
               if Dbg and then Lr = 1 then Dump ("  kqv p=" & Pos'Image & " L0", Ctx_O); end if;
               Add_To (H, GMV (B.W_O, Ctx_O));   -- attn residual
               if Dbg then Dump ("  attn_resid p=" & Pos'Image & " L" & Integer'Image (Lr - 1), H); end if;
            end;

            --  SwiGLU FFN with residual.
            declare
               T_Ffn : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
               Xf   : constant Tensor := GN (H, B.Ffn_Norm);
               Gate : constant Tensor := Silu (GMV (B.W_Gate, Xf));
               Up   : constant Tensor := GMV (B.W_Up, Xf);
            begin
               Add_To (H, GMV (B.W_Down, Gate * Up));
               if Prof then
                  Acc_Ffn := Acc_Ffn +
                    Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - T_Ffn);
               end if;
            end;
            if Dbg then Dump ("  l_out p=" & Pos'Image & " L" & Integer'Image (Lr - 1), H); end if;
         end;
      end loop;

      declare
         R : constant Tensor := GMV (M.Output, GN (H, M.Out_Norm));
      begin
         if Prof then
            Acc_Total := Acc_Total +
              Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - T_Step);
         end if;
         return R;
      end;
   end Forward_Step;

   --------------------------------------------------------------------
   -- Batched forward: advance B sequences by one token in ONE pass.
   -- Projections/FFN/output go through the batched GPU matmul (weight read
   -- once, reused across B); attention stays per-sequence (each its own KV
   -- cache + position), parallelized over (sequence x head). This is the
   -- continuous-batching compute primitive (the scheduler drives it).
   --------------------------------------------------------------------

   type KV_Cache_Ptr is access KV_Cache;
   type Seq_Cache_Array is array (Positive range <>) of KV_Cache_Ptr;
   type Int_Array is array (Positive range <>) of Integer;
   type Tensor_List is array (Positive range <>) of Tensor;

   --  Y[B,Out] = X[B,In] . W. GPU batched matmul when available, else B matvecs.
   function GMV_Batch (W : LLM_Weight.Weight; X : Tensor; B : Integer)
      return Tensor
   is
      KC   : constant Integer := LLM_Weight.Kind_Code (W);
      Ind  : constant Integer := LLM_Weight.Cols (W);
      Outd : constant Integer := LLM_Weight.Rows (W);
   begin
      return Y : Tensor := New_Tensor ([B, Outd]) do
         if KC >= 0 and then LLM_GPU.Available and then LLM_GPU.Has_MatMul then
            LLM_GPU.MatMul
              (LLM_Weight.Raw_Address (W), LLM_Weight.Raw_Bytes (W),
               KC, Ind, Outd, B, Data_Address (X), Data_Address (Y));
         else
            for Bi in 0 .. B - 1 loop
               declare
                  Yb : constant Tensor := GMV (W, Slice (X, Bi * Ind + 1, Ind));
               begin
                  for O in 1 .. Outd loop
                     Set_Flat (Y, Bi * Outd + O, Get_Flat (Yb, O));
                  end loop;
               end;
            end loop;
         end if;
      end return;
   end GMV_Batch;

   --  Row-wise RMSNorm of a [B, D] tensor (one normalization per row).
   function RMSNorm_Batch (H : Tensor; Wt : Tensor; B, D : Integer) return Tensor is
   begin
      return R : Tensor := New_Tensor ([B, D]) do
         for Bi in 0 .. B - 1 loop
            declare
               Base : constant Integer := Bi * D;
               Ss   : Float := 0.0;
            begin
               for I in 1 .. D loop
                  declare V : constant Float := Get_Flat (H, Base + I); begin
                     Ss := Ss + V * V;
                  end;
               end loop;
               declare
                  Rms : constant Float := Sqrt (Ss / Float (D) + RMS_Eps);
               begin
                  for I in 1 .. D loop
                     Set_Flat (R, Base + I,
                       (Get_Flat (H, Base + I) / Rms) * Get_Flat (Wt, I));
                  end loop;
               end;
            end;
         end loop;
      end return;
   end RMSNorm_Batch;

   procedure Forward_Batch
     (M         : Llama_Model;
      Seqs      : Seq_Cache_Array;   -- 1 .. B, each a per-layer KV cache
      Toks      : Int_Array;         -- 1 .. B input tokens
      Positions : Int_Array;         -- 1 .. B current 0-based positions
      Logits    : out Tensor_List)   -- 1 .. B next-token logit rows [1, Vocab]
   is
      D   : constant Integer := M.Dim;
      NH  : constant Integer := M.N_Heads;
      NKV : constant Integer := M.N_KV;
      HD  : constant Integer := M.Head_Dim;
      AScale : constant Float := 1.0 / Sqrt (Float (HD));
      B   : constant Integer := Seqs'Length;
      QW  : constant Integer := NH * HD;    -- query row width
      KW  : constant Integer := NKV * HD;   -- kv row width

      H : Tensor := New_Tensor ([B, D]);    -- batched residual
   begin
      --  Gather each sequence's input-token embedding into its row.
      for Bi in 1 .. B loop
         declare
            E : constant Tensor := LLM_Weight.Get_Row (M.Tok_Emb, Toks (Bi));
         begin
            for I in 1 .. D loop
               Set_Flat (H, (Bi - 1) * D + I, Get_Flat (E, I));
            end loop;
         end;
      end loop;

      for Lr in 1 .. M.N_Blocks loop
         declare
            Blk : L_Block renames M.Blocks (Lr);
            X   : constant Tensor := RMSNorm_Batch (H, Blk.Attn_Norm, B, D);
            Q   : Tensor := GMV_Batch (Blk.W_Q, X, B);            -- [B, QW]
            K   : constant Tensor := GMV_Batch (Blk.W_K, X, B);   -- [B, KW]
            V   : constant Tensor := GMV_Batch (Blk.W_V, X, B);   -- [B, KW]
            Ctx : Tensor := New_Tensor ([B, QW]);
         begin
            --  Per sequence: RoPE Q/K at its position, append K/V to its cache.
            for Bi in 1 .. B loop
               declare
                  Pos  : constant Integer := Positions (Bi);
                  QB   : constant Integer := (Bi - 1) * QW;
                  KB   : constant Integer := (Bi - 1) * KW;
                  Krow : Tensor := New_Tensor ([1, KW]);
                  Vrow : Tensor := New_Tensor ([1, KW]);
               begin
                  for Hh in 0 .. NH - 1 loop
                     declare
                        S : constant Tensor := LLM_RoPE.Apply
                          (M.RoPE, Slice (Q, QB + Hh * HD + 1, HD), Pos);
                     begin
                        for J in 1 .. HD loop
                           Set_Flat (Q, QB + Hh * HD + J, Get_Flat (S, J));
                        end loop;
                     end;
                  end loop;
                  for Hh in 0 .. NKV - 1 loop
                     declare
                        S : constant Tensor := LLM_RoPE.Apply
                          (M.RoPE, Slice (K, KB + Hh * HD + 1, HD), Pos);
                     begin
                        for J in 1 .. HD loop
                           Set_Flat (Krow, Hh * HD + J, Get_Flat (S, J));
                        end loop;
                     end;
                  end loop;
                  for J in 1 .. KW loop
                     Set_Flat (Vrow, J, Get_Flat (V, KB + J));
                  end loop;
                  Seqs (Bi) (Lr).K (Pos + 1) := Krow;
                  Seqs (Bi) (Lr).V (Pos + 1) := Vrow;
               end;
            end loop;

            --  Attention over (sequence x head); each job writes a disjoint
            --  Ctx slice and only reads Q + that sequence's cache.
            declare
               type BHead_Op is new LLM_Pool.Parallel_Op with null record;
               overriding procedure Execute
                 (Op : in out BHead_Op; Lo, Hi : Integer)
               is
               begin
                  for Job in Lo .. Hi loop
                     declare
                        Bi   : constant Integer := Job / NH;          -- 0-based seq
                        Hh   : constant Integer := Job mod NH;        -- head
                        Pos  : constant Integer := Positions (Bi + 1);
                        KVh  : constant Integer := Hh / (NH / NKV);
                        QB   : constant Integer := Bi * QW + Hh * HD;
                        KC   : Tensor_Array_Ptr renames Seqs (Bi + 1) (Lr).K;
                        VC   : Tensor_Array_Ptr renames Seqs (Bi + 1) (Lr).V;
                        Scr  : array (0 .. Pos) of Float;
                        Mx   : Float := Float'First;
                        Den  : Float := 0.0;
                     begin
                        for S in 0 .. Pos loop
                           declare Dp : Float := 0.0; begin
                              for J in 1 .. HD loop
                                 Dp := Dp + Get_Flat (Q, QB + J)
                                          * Get_Flat (KC (S + 1), KVh * HD + J);
                              end loop;
                              Scr (S) := Dp * AScale;
                              Mx := Float'Max (Mx, Scr (S));
                           end;
                        end loop;
                        for S in 0 .. Pos loop
                           Scr (S) := Exp (Scr (S) - Mx); Den := Den + Scr (S);
                        end loop;
                        for J in 1 .. HD loop
                           declare Acc : Float := 0.0; begin
                              for S in 0 .. Pos loop
                                 Acc := Acc + (Scr (S) / Den)
                                          * Get_Flat (VC (S + 1), KVh * HD + J);
                              end loop;
                              Set_Flat (Ctx, Bi * QW + Hh * HD + J, Acc);
                           end;
                        end loop;
                     end;
                  end loop;
               end Execute;
               AOp : BHead_Op;
            begin
               LLM_Pool.Run (AOp, 0, B * NH - 1, Min_Grain => 2);
            end;

            --  O projection + attention residual.
            declare
               O : constant Tensor := GMV_Batch (Blk.W_O, Ctx, B);
            begin
               for I in 1 .. B * D loop
                  Set_Flat (H, I, Get_Flat (H, I) + Get_Flat (O, I));
               end loop;
            end;

            --  SwiGLU FFN + residual, batched.
            declare
               Xf   : constant Tensor := RMSNorm_Batch (H, Blk.Ffn_Norm, B, D);
               Gate : constant Tensor := Silu (GMV_Batch (Blk.W_Gate, Xf, B));
               Up   : constant Tensor := GMV_Batch (Blk.W_Up, Xf, B);
               Down : constant Tensor := GMV_Batch (Blk.W_Down, Gate * Up, B);
            begin
               for I in 1 .. B * D loop
                  Set_Flat (H, I, Get_Flat (H, I) + Get_Flat (Down, I));
               end loop;
            end;
         end;
      end loop;

      --  Final norm + output projection, then split into per-sequence rows.
      declare
         HN : constant Tensor := RMSNorm_Batch (H, M.Out_Norm, B, D);
         L  : constant Tensor := GMV_Batch (M.Output, HN, B);   -- [B, Vocab]
         Vc : constant Integer := M.Vocab;
      begin
         for Bi in 1 .. B loop
            Logits (Bi) := Slice (L, (Bi - 1) * Vc + 1, Vc);
         end loop;
      end;
   end Forward_Batch;

   function Batch_Self_Test (M : Llama_Model) return Float is
      Seq1 : constant Int_Array := [M.Bos, 9906, 1879, 11];   -- arbitrary valid ids
      Seq2 : constant Int_Array := [M.Bos, 40, 1097, 13];
      Len  : constant Integer := Seq1'Length;
      Cap  : constant Integer := Len + 1;

      procedure Free is
        new Ada.Unchecked_Deallocation (Tensor_Array, Tensor_Array_Ptr);

      function New_Cache return KV_Cache_Ptr is
         C : constant KV_Cache_Ptr := new KV_Cache (1 .. M.N_Blocks);
      begin
         for L in 1 .. M.N_Blocks loop
            C (L).K := new Tensor_Array (1 .. Cap);
            C (L).V := new Tensor_Array (1 .. Cap);
         end loop;
         return C;
      end New_Cache;

      procedure Release (C : KV_Cache_Ptr) is
      begin
         for L in 1 .. M.N_Blocks loop Free (C (L).K); Free (C (L).V); end loop;
      end Release;

      --  Single-path final logits for one sequence.
      function Single_Logits (S : Int_Array) return Tensor is
         C  : constant KV_Cache_Ptr := New_Cache;
         Lg : Tensor;
      begin
         for I in S'Range loop
            Lg := Forward_Step (M, C.all, S (I), I - S'First);
         end loop;
         Release (C);
         return Lg;
      end Single_Logits;

      Ref1 : constant Tensor := Single_Logits (Seq1);
      Ref2 : constant Tensor := Single_Logits (Seq2);
      C1 : constant KV_Cache_Ptr := New_Cache;
      C2 : constant KV_Cache_Ptr := New_Cache;
      Seqs : constant Seq_Cache_Array := [C1, C2];
      BL   : Tensor_List (1 .. 2);
      Diff : Float := 0.0;
   begin
      for T in 0 .. Len - 1 loop
         Forward_Batch
           (M, Seqs,
            Toks      => [Seq1 (Seq1'First + T), Seq2 (Seq2'First + T)],
            Positions => [T, T],
            Logits    => BL);
      end loop;
      for I in 1 .. M.Vocab loop
         Diff := Float'Max (Diff, abs (Get_Flat (BL (1), I) - Get_Flat (Ref1, I)));
         Diff := Float'Max (Diff, abs (Get_Flat (BL (2), I) - Get_Flat (Ref2, I)));
      end loop;
      Release (C1); Release (C2);
      return Diff;
   end Batch_Self_Test;

   --------------------------------------------------------------------
   -- Continuous-batch scheduler: drive N sequences to completion in ONE
   -- interleaved loop. Each step gathers the still-running sequences, runs a
   -- single batched forward, then per sequence either advances its prefill or
   -- samples + streams its next token. The batch shrinks as sequences finish
   -- (true continuous batching; dynamic admission is layered on in the server).
   --------------------------------------------------------------------

   type Tok_Arr_Acc is access constant LLM_Tokenizer.Token_Array;
   type Prompt_Arr  is array (Positive range <>) of Tok_Arr_Acc;
   type UStr_Arr    is array (Positive range <>) of Unbounded_String;
   type Sink_Acc    is access all LLM_Qwen.Token_Sink'Class;
   type Sink_Arr    is array (Positive range <>) of Sink_Acc;

   procedure Generate_Batch
     (M       : Llama_Model;
      Prompts : Prompt_Arr;
      Max_New : Integer;
      Stop_A, Stop_B : Integer;
      Sinks   : Sink_Arr;        -- 'Range = Prompts'Range; entries may be null
      Results : out UStr_Arr;
      Params  : LLM_Sampler.Params := LLM_Sampler.Greedy)
   is
      N : constant Integer := Prompts'Length;
      type Phase_T is (Ph_Prefill, Ph_Decode, Ph_Done);
      type Slot is record
         Cache  : KV_Cache_Ptr;
         PFirst : Integer;          -- Prompt'First (for indexing)
         PLen   : Integer;
         Pos    : Integer := 0;
         In_Tok : Integer := 0;
         Ph     : Phase_T := Ph_Prefill;
         Smp    : LLM_Sampler.Sampler := LLM_Sampler.Create (Params);
         Txt    : Unbounded_String;
         N_Gen  : Integer := 0;
         Cap    : Integer;
         Sink   : Sink_Acc;
      end record;
      Slots  : array (1 .. N) of Slot;
      Active : Integer := N;
      Empty_Hist : constant LLM_Sampler.History (1 .. 0) := [others => 0];

      procedure Free is
        new Ada.Unchecked_Deallocation (Tensor_Array, Tensor_Array_Ptr);
      function New_Cache (Cap : Integer) return KV_Cache_Ptr is
         C : constant KV_Cache_Ptr := new KV_Cache (1 .. M.N_Blocks);
      begin
         for L in 1 .. M.N_Blocks loop
            C (L).K := new Tensor_Array (1 .. Cap);
            C (L).V := new Tensor_Array (1 .. Cap);
         end loop;
         return C;
      end New_Cache;
   begin
      for I in 1 .. N loop
         declare
            P : constant Tok_Arr_Acc := Prompts (Prompts'First + I - 1);
         begin
            Slots (I).PFirst := P'First;
            Slots (I).PLen   := P'Length;
            Slots (I).Cap    := P'Length + Max_New + 1;
            Slots (I).Cache  := New_Cache (Slots (I).Cap);
            Slots (I).In_Tok := P (P'First);
            Slots (I).Sink   := Sinks (Sinks'First + I - 1);
         end;
      end loop;

      while Active > 0 loop
         declare
            B   : Integer := 0;
            Cs  : Seq_Cache_Array (1 .. Active);
            Tk  : Int_Array (1 .. Active);
            Ps  : Int_Array (1 .. Active);
            Mp  : Int_Array (1 .. Active);
            Lg  : Tensor_List (1 .. Active);
         begin
            for S in 1 .. N loop
               if Slots (S).Ph /= Ph_Done then
                  B := B + 1;
                  Cs (B) := Slots (S).Cache;
                  Tk (B) := Slots (S).In_Tok;
                  Ps (B) := Slots (S).Pos;
                  Mp (B) := S;
               end if;
            end loop;

            Forward_Batch (M, Cs (1 .. B), Tk (1 .. B), Ps (1 .. B), Lg (1 .. B));

            for Bi in 1 .. B loop
               declare
                  S : constant Integer := Mp (Bi);
                  P : constant Tok_Arr_Acc :=
                    Prompts (Prompts'First + S - 1);
               begin
                  if Slots (S).Ph = Ph_Prefill
                     and then Slots (S).Pos < Slots (S).PLen - 1
                  then
                     --  Still feeding the prompt: advance to the next prompt token.
                     Slots (S).Pos := Slots (S).Pos + 1;
                     Slots (S).In_Tok := P (Slots (S).PFirst + Slots (S).Pos);
                  else
                     --  Last prompt token reached (or already decoding): sample.
                     declare
                        Tid : constant Integer :=
                          LLM_Sampler.Next (Slots (S).Smp, Lg (Bi), Empty_Hist);
                     begin
                        if Tid = M.Eos or else Tid = M.Eot
                           or else Tid = Stop_A or else Tid = Stop_B
                        then
                           Slots (S).Ph := Ph_Done; Active := Active - 1;
                        else
                           declare
                              Piece : constant String :=
                                LLM_Tokenizer.Decode_One (M.Tok, Tid);
                           begin
                              Append (Slots (S).Txt, Piece);
                              if Slots (S).Sink /= null then
                                 LLM_Qwen.Emit (Slots (S).Sink.all, Piece);
                              end if;
                           end;
                           Slots (S).In_Tok := Tid;
                           Slots (S).Pos    := Slots (S).Pos + 1;
                           Slots (S).N_Gen  := Slots (S).N_Gen + 1;
                           Slots (S).Ph     := Ph_Decode;
                           if Slots (S).N_Gen >= Max_New
                              or else Slots (S).Pos >= Slots (S).Cap
                           then
                              Slots (S).Ph := Ph_Done; Active := Active - 1;
                           end if;
                        end if;
                     end;
                  end if;
               end;
            end loop;
         end;
      end loop;

      for I in 1 .. N loop
         Results (Results'First + I - 1) := Slots (I).Txt;
         for L in 1 .. M.N_Blocks loop
            Free (Slots (I).Cache (L).K); Free (Slots (I).Cache (L).V);
         end loop;
      end loop;
   end Generate_Batch;

   function Batch_Gen_Self_Test (M : Llama_Model; Max_New : Integer) return String
   is
      use type LLM_Tokenizer.Token_Array;
      P1 : aliased constant LLM_Tokenizer.Token_Array :=
        LLM_Tokenizer.Token_Array'(1 => M.Bos)
          & LLM_Tokenizer.Encode (M.Tok, "The capital of France is");
      P2 : aliased constant LLM_Tokenizer.Token_Array :=
        LLM_Tokenizer.Token_Array'(1 => M.Bos)
          & LLM_Tokenizer.Encode (M.Tok, "Roses are red, violets are");
      Single1 : constant String := Generate (M, P1, -1, -1, Max_New);
      Single2 : constant String := Generate (M, P2, -1, -1, Max_New);
      Res     : UStr_Arr (1 .. 2);
      Ok1, Ok2 : Boolean;
   begin
      Generate_Batch
        (M, [P1'Unchecked_Access, P2'Unchecked_Access], Max_New, -1, -1,
         [null, null], Res);
      Ok1 := To_String (Res (1)) = Single1;
      Ok2 := To_String (Res (2)) = Single2;
      return "seq1 match=" & Boolean'Image (Ok1)
        & " seq2 match=" & Boolean'Image (Ok2)
        & ASCII.LF & "  single1: '" & Single1 & "'"
        & ASCII.LF & "  batched1: '" & To_String (Res (1)) & "'"
        & ASCII.LF & "  single2: '" & Single2 & "'"
        & ASCII.LF & "  batched2: '" & To_String (Res (2)) & "'";
   end Batch_Gen_Self_Test;

   --------------------------------------------------------------------
   -- Continuous-batch SERVER SCHEDULER. One task owns the batched forward;
   -- handler tasks Run_Request (enqueue + block, streaming via their Sink).
   -- Many sessions share each forward step → real concurrent throughput.
   -- Only this task touches the GPU/pool, so no per-step lock is needed.
   --------------------------------------------------------------------

   Sched_Max_Seq : constant := 8;       -- concurrent sequences per batch
   Sched_Cap     : constant := 4096;    -- KV positions per slot (prompt + gen)

   type Sched_Tok_Ptr is access LLM_Tokenizer.Token_Array;
   type Sink_Ptr is access all LLM_Qwen.Token_Sink'Class;
   function To_Sink is new Ada.Unchecked_Conversion (System.Address, Sink_Ptr);

   type Request is limited record
      Prompt    : Sched_Tok_Ptr;
      Max       : Integer;
      Stop_A, Stop_B : Integer;
      Params    : LLM_Sampler.Params;
      --  Sink object address (not a typed access): the sink is a handler-local
      --  object and the handler BLOCKS until this request finishes, so it stays
      --  alive — but Ada's accessibility rules forbid storing the access in this
      --  longer-lived record, so we carry the address and rebuild the pointer.
      Sink_Addr : System.Address;
      Result    : Unbounded_String;
      Done      : Ada.Synchronous_Task_Control.Suspension_Object;
   end record;
   type Request_Acc is access all Request;
   type Req_Ring is array (1 .. 64) of Request_Acc;

   protected Req_Queue is
      entry Put (R : Request_Acc);
      entry Get_Wait (R : out Request_Acc);
      procedure Try_Get (R : out Request_Acc; Got : out Boolean);
   private
      Buf : Req_Ring := [others => null];
      Cnt : Natural := 0;
      Hd  : Positive := 1;
      Tl  : Positive := 1;
   end Req_Queue;

   protected body Req_Queue is
      entry Put (R : Request_Acc) when Cnt < Buf'Length is
      begin
         Buf (Tl) := R; Tl := Tl mod Buf'Length + 1; Cnt := Cnt + 1;
      end Put;
      entry Get_Wait (R : out Request_Acc) when Cnt > 0 is
      begin
         R := Buf (Hd); Hd := Hd mod Buf'Length + 1; Cnt := Cnt - 1;
      end Get_Wait;
      procedure Try_Get (R : out Request_Acc; Got : out Boolean) is
      begin
         if Cnt > 0 then
            R := Buf (Hd); Hd := Hd mod Buf'Length + 1; Cnt := Cnt - 1; Got := True;
         else
            R := null; Got := False;
         end if;
      end Try_Get;
   end Req_Queue;

   --  One-shot, race-free lazy start: the caller that gets Do_It=True starts
   --  the scheduler task (Sched.Init can't be called from inside a protected).
   protected Init_Guard is
      procedure Claim (Do_It : out Boolean);
   private
      Claimed : Boolean := False;
   end Init_Guard;
   protected body Init_Guard is
      procedure Claim (Do_It : out Boolean) is
      begin Do_It := not Claimed; Claimed := True; end Claim;
   end Init_Guard;

   task Sched is
      entry Init (Mdl : Llama_Model);
   end Sched;

   task body Sched is
      M : Llama_Model;
      type Phase_T is (Ph_Free, Ph_Prefill, Ph_Decode);
      type SSlot is record
         Cache  : KV_Cache_Ptr;
         Req    : Request_Acc := null;
         PFirst, PLen, Pos, N_Gen : Integer := 0;
         In_Tok : Integer := 0;
         Ph     : Phase_T := Ph_Free;
         Smp    : LLM_Sampler.Sampler;
         Hist   : LLM_Sampler.History (1 .. Sched_Cap) := [others => 0];
         N_Hist : Natural := 0;
      end record;
      Slots  : array (1 .. Sched_Max_Seq) of SSlot;
      Active : Natural := 0;

      function New_Cache return KV_Cache_Ptr is
         C : constant KV_Cache_Ptr := new KV_Cache (1 .. M.N_Blocks);
      begin
         for L in 1 .. M.N_Blocks loop
            C (L).K := new Tensor_Array (1 .. Sched_Cap);
            C (L).V := new Tensor_Array (1 .. Sched_Cap);
         end loop;
         return C;
      end New_Cache;

      procedure Admit (R : Request_Acc) is
      begin
         for S in Slots'Range loop
            if Slots (S).Ph = Ph_Free then
               Slots (S).Req    := R;
               Slots (S).PFirst := R.Prompt'First;
               Slots (S).PLen   := R.Prompt'Length;
               Slots (S).Pos    := 0;
               Slots (S).N_Gen  := 0;
               Slots (S).N_Hist := 0;
               Slots (S).In_Tok := R.Prompt (R.Prompt'First);
               Slots (S).Smp    := LLM_Sampler.Create (R.Params);
               Slots (S).Ph     := Ph_Prefill;
               Active := Active + 1;
               return;
            end if;
         end loop;
      end Admit;

      procedure Retire (S : Integer) is
      begin
         Ada.Synchronous_Task_Control.Set_True (Slots (S).Req.Done);
         Slots (S).Req := null;
         Slots (S).Ph  := Ph_Free;
         Active := Active - 1;
      end Retire;
   begin
      accept Init (Mdl : Llama_Model) do M := Mdl; end Init;
      for S in Slots'Range loop Slots (S).Cache := New_Cache; end loop;

      loop
       begin
         --  Block for the first request when idle; then drain the queue into
         --  any free slots so newly-arrived sessions join the running batch.
         if Active = 0 then
            declare R : Request_Acc; begin
               Req_Queue.Get_Wait (R); Admit (R);
            end;
         end if;
         while Active < Sched_Max_Seq loop
            declare R : Request_Acc; Got : Boolean; begin
               Req_Queue.Try_Get (R, Got);
               exit when not Got;
               Admit (R);
            end;
         end loop;

         --  One batched forward over all active slots.
         declare
            B  : Integer := 0;
            Cs : Seq_Cache_Array (1 .. Active);
            Tk : Int_Array (1 .. Active);
            Ps : Int_Array (1 .. Active);
            Mp : Int_Array (1 .. Active);
            Lg : Tensor_List (1 .. Active);
         begin
            for S in Slots'Range loop
               if Slots (S).Ph /= Ph_Free then
                  B := B + 1;
                  Cs (B) := Slots (S).Cache;
                  Tk (B) := Slots (S).In_Tok;
                  Ps (B) := Slots (S).Pos;
                  Mp (B) := S;
               end if;
            end loop;

            Forward_Batch (M, Cs (1 .. B), Tk (1 .. B), Ps (1 .. B), Lg (1 .. B));

            for Bi in 1 .. B loop
               declare
                  S : constant Integer := Mp (Bi);
                  R : constant Request_Acc := Slots (S).Req;
               begin
                  if Slots (S).Ph = Ph_Prefill
                     and then Slots (S).Pos < Slots (S).PLen - 1
                  then
                     Slots (S).Pos := Slots (S).Pos + 1;
                     Slots (S).In_Tok := R.Prompt (Slots (S).PFirst + Slots (S).Pos);
                  else
                     declare
                        Win : constant Natural := Integer'Min
                          (Slots (S).N_Hist,
                           Integer'Max (0, R.Params.Repeat_Last_N));
                        Tid : constant Integer := LLM_Sampler.Next
                          (Slots (S).Smp, Lg (Bi),
                           Slots (S).Hist (Slots (S).N_Hist - Win + 1 .. Slots (S).N_Hist));
                     begin
                        if Tid = M.Eos or else Tid = M.Eot
                           or else Tid = R.Stop_A or else Tid = R.Stop_B
                        then
                           Retire (S);
                        else
                           declare
                              Piece : constant String :=
                                LLM_Tokenizer.Decode_One (M.Tok, Tid);
                           begin
                              Append (R.Result, Piece);
                              if R.Sink_Addr /= System.Null_Address then
                                 LLM_Qwen.Emit (To_Sink (R.Sink_Addr).all, Piece);
                              end if;
                           end;
                           Slots (S).N_Hist := Slots (S).N_Hist + 1;
                           Slots (S).Hist (Slots (S).N_Hist) := Tid;
                           Slots (S).In_Tok := Tid;
                           Slots (S).Pos    := Slots (S).Pos + 1;
                           Slots (S).N_Gen  := Slots (S).N_Gen + 1;
                           Slots (S).Ph     := Ph_Decode;
                           if Slots (S).N_Gen >= R.Max
                              or else Slots (S).Pos >= Sched_Cap
                           then
                              Retire (S);
                           end if;
                        end if;
                     end;
                  end if;
               exception
                  --  Isolate a per-client fault (e.g. the client disconnected
                  --  mid-stream → Emit raises on the dead socket): drop just
                  --  this slot, unblock its handler, keep the batch running.
                  when others =>
                     Retire (S);
               end;
            end loop;
         end;
       exception
         --  A fault in one step must NOT kill the scheduler (that would hang
         --  every client forever). Log it, unblock all in-flight requests
         --  (they get whatever partial text accumulated), and carry on.
         when E : others =>
            Ada.Text_IO.Put_Line
              ("  [scheduler] step fault, recovering: "
               & Ada.Exceptions.Exception_Name (E));
            for S in Slots'Range loop
               if Slots (S).Ph /= Ph_Free then
                  Ada.Synchronous_Task_Control.Set_True (Slots (S).Req.Done);
                  Slots (S).Req := null;
                  Slots (S).Ph  := Ph_Free;
               end if;
            end loop;
            Active := 0;
       end;
      end loop;
   end Sched;

   function Run_Request
     (M : Llama_Model; Prompt : LLM_Tokenizer.Token_Array;
      Max, Stop_A, Stop_B : Integer;
      Sink : access LLM_Qwen.Token_Sink'Class;
      Params : LLM_Sampler.Params) return String
   is
      procedure Free_R is new Ada.Unchecked_Deallocation (Request, Request_Acc);
      procedure Free_T is
        new Ada.Unchecked_Deallocation (LLM_Tokenizer.Token_Array, Sched_Tok_Ptr);
      Do_It : Boolean;
      R     : Request_Acc := new Request;
   begin
      Init_Guard.Claim (Do_It);
      if Do_It then Sched.Init (M); end if;
      --  Clamp so prompt + generation fit the slot's KV cache (a long
      --  multi-turn prompt would otherwise overflow). Keep the TAIL (most
      --  recent context + the trailing assistant header).
      declare
         Room : constant Integer := Sched_Cap - Integer'Max (1, Max) - 2;
      begin
         if Prompt'Length > Room and then Room > 0 then
            R.Prompt := new LLM_Tokenizer.Token_Array'
              (Prompt (Prompt'Last - Room + 1 .. Prompt'Last));
         else
            R.Prompt := new LLM_Tokenizer.Token_Array'(Prompt);
         end if;
      end;
      R.Max := Max; R.Stop_A := Stop_A; R.Stop_B := Stop_B;
      R.Params := Params;
      R.Sink_Addr := (if Sink /= null then Sink.all'Address
                      else System.Null_Address);
      Ada.Synchronous_Task_Control.Set_False (R.Done);
      Req_Queue.Put (R);
      Ada.Synchronous_Task_Control.Suspend_Until_True (R.Done);
      return Res : constant String := To_String (R.Result) do
         Free_T (R.Prompt);
         Free_R (R);
      end return;
   end Run_Request;

   --------------------------------------------------------------------
   -- Greedy decode with incremental K/V cache.
   --------------------------------------------------------------------

   function Generate
     (M : Llama_Model; Ids : LLM_Tokenizer.Token_Array;
      Stop_A, Stop_B : Integer := -1;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String
   is
      Cap    : constant Integer := Integer'Max (1, Ids'Length + Max_New_Tokens);
      Cache  : KV_Cache (1 .. M.N_Blocks);
      Len    : Integer := 0;
      Out_S  : Unbounded_String;
      Logits : Tensor;
      Smp    : LLM_Sampler.Sampler := LLM_Sampler.Create (Params);
      Hist   : LLM_Sampler.History (1 .. Integer'Max (1, Max_New_Tokens)) :=
        [others => 0];
      N_Hist : Natural := 0;

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
      for L in 1 .. M.N_Blocks loop
         Cache (L).K := new Tensor_Array (1 .. Cap);
         Cache (L).V := new Tensor_Array (1 .. Cap);
      end loop;

      for I in Ids'Range loop
         Logits := Locked_Step (Ids (I), Len);
         Len := Len + 1;
      end loop;

      for Step in 1 .. Max_New_Tokens loop
         declare
            Win : constant Natural :=
              Integer'Min (N_Hist, Integer'Max (0, Params.Repeat_Last_N));
            Tid : constant Integer := LLM_Sampler.Next
              (Smp, Logits, Hist (N_Hist - Win + 1 .. N_Hist));
         begin
            exit when Tid = M.Eos or else Tid = M.Eot
              or else Tid = Stop_A or else Tid = Stop_B;
            declare Piece : constant String := LLM_Tokenizer.Decode_One (M.Tok, Tid); begin
               Append (Out_S, Piece);
               if Sink /= null then LLM_Qwen.Emit (Sink.all, Piece); end if;
            end;
            N_Hist := N_Hist + 1; Hist (N_Hist) := Tid;
            exit when Len >= Cap;
            Logits := Locked_Step (Tid, Len);
            Len := Len + 1;
         end;
      end loop;

      for L in 1 .. M.N_Blocks loop
         Free (Cache (L).K); Free (Cache (L).V);
      end loop;
      if Prof then
         Ada.Text_IO.Put_Line
           ("PROF total=" & Duration'Image (Acc_Total)
            & "  matvec=" & Duration'Image (Acc_Mv)
            & "  attn=" & Duration'Image (Acc_Attn)
            & "  rope=" & Duration'Image (Acc_Rope)
            & "  ffn-blk(incl mv)=" & Duration'Image (Acc_Ffn)
            & "  other=" & Duration'Image
                (Acc_Total - Acc_Attn - Acc_Mv - Acc_Rope) & "s");
         Acc_Total := 0.0; Acc_Attn := 0.0; Acc_Mv := 0.0;
         Acc_Rope := 0.0; Acc_Ffn := 0.0;
      end if;
      return To_String (Out_S);
   end Generate;

   --------------------------------------------------------------------
   -- Llama-3 header chat template.
   --------------------------------------------------------------------

   function Chat
     (M : Llama_Model; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String
   is
      use type LLM_Tokenizer.Token_Array;
      LF : constant Character := Character'Val (10);

      function One (Id : Integer) return LLM_Tokenizer.Token_Array is
        (if Id >= 0 then LLM_Tokenizer.Token_Array'(1 => Id)
         else LLM_Tokenizer.Token_Array'(2 .. 1 => 0));

      --  <|start_header_id|>{role}<|end_header_id|>\n\n
      function Header (Role : String) return LLM_Tokenizer.Token_Array is
        (One (M.SH) & LLM_Tokenizer.Encode (M.Tok, Role)
           & One (M.EH) & LLM_Tokenizer.Encode (M.Tok, LF & LF));

      function Msg_Ids (Msg : LLM_Qwen.Message) return LLM_Tokenizer.Token_Array is
         Role : constant String :=
           (case Msg.Role is
              when LLM_Qwen.Role_System    => "system",
              when LLM_Qwen.Role_User      => "user",
              when LLM_Qwen.Role_Assistant => "assistant");
      begin
         return Header (Role)
           & LLM_Tokenizer.Encode (M.Tok, To_String (Msg.Text)) & One (M.Eot);
      end Msg_Ids;

      function Conv_Ids (I : Positive) return LLM_Tokenizer.Token_Array is
      begin
         if I > Conversation'Last then return LLM_Tokenizer.Token_Array'(2 .. 1 => 0); end if;
         return Msg_Ids (Conversation (I)) & Conv_Ids (I + 1);
      end Conv_Ids;
   begin
      --  Route through the continuous-batch scheduler so concurrent chat
      --  sessions share batched forward passes (Generate is the single-stream
      --  path, kept for Complete / direct callers).
      return Run_Request
        (M, One (M.Bos) & Conv_Ids (Conversation'First) & Header ("assistant"),
         Max_New_Tokens, -1, -1, Sink, Params);
   end Chat;

   function Complete
     (M : Llama_Model; Prompt : String; Max_New_Tokens : Integer := 8)
      return String
   is
      use type LLM_Tokenizer.Token_Array;
   begin
      return Generate
        (M, LLM_Tokenizer.Token_Array'(1 => M.Bos)
              & LLM_Tokenizer.Encode (M.Tok, Prompt),
         -1, -1, Max_New_Tokens, null);
   end Complete;

   function Vocab_Size  (M : Llama_Model) return Integer is (M.Vocab);
   function Dim         (M : Llama_Model) return Integer is (M.Dim);
   function Block_Count (M : Llama_Model) return Integer is (M.N_Blocks);

end LLM_Llama;
