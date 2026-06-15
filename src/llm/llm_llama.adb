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
with LLM_GGUF;    use LLM_GGUF;
with LLM_Tensor;  use LLM_Tensor;
with LLM_RMSNorm;
with LLM_RoPE;
with LLM_Weight;

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

   function GN (X, W : Tensor) return Tensor is (LLM_RMSNorm.Forward (X, W));

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
      if M.Has_Freqs then LLM_RoPE.Set_Freq_Factors (M.RoPE, M.Rope_Freqs); end if;

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
   begin
      for Lr in 1 .. M.N_Blocks loop
         declare
            B : L_Block renames M.Blocks (Lr);
            X : constant Tensor := GN (H, B.Attn_Norm);
            Q : Tensor := LLM_Weight.MatVec (B.W_Q, X);
            K : Tensor := LLM_Weight.MatVec (B.W_K, X);
            V : constant Tensor := LLM_Weight.MatVec (B.W_V, X);
         begin
            --  RoPE on Q (per head) and K (per kv head); no QK-norm, no bias.
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
            Cache (Lr).K (Pos + 1) := K;
            Cache (Lr).V (Pos + 1) := V;

            --  Causal GQA attention over cached positions; scale 1/sqrt(HD).
            declare
               KC    : Tensor_Array_Ptr renames Cache (Lr).K;
               VC    : Tensor_Array_Ptr renames Cache (Lr).V;
               Ctx_O : Tensor := New_Tensor ([1, NH * HD]);
            begin
               for Hh in 0 .. NH - 1 loop
                  declare
                     KV  : constant Integer := Hh / (NH / NKV);
                     Scr : Tensor := New_Tensor ([1, Pos + 1]);
                     Mx  : Float := Float'First;
                     Den : Float := 0.0;
                  begin
                     for S in 0 .. Pos loop
                        declare Dp : Float := 0.0; begin
                           for J in 1 .. HD loop
                              Dp := Dp + Get_Flat (Q, Hh * HD + J)
                                       * Get_Flat (KC (S + 1), KV * HD + J);
                           end loop;
                           Set_Flat (Scr, S + 1, Dp * AScale);
                           Mx := Float'Max (Mx, Dp * AScale);
                        end;
                     end loop;
                     for S in 0 .. Pos loop
                        Set_Flat (Scr, S + 1, Exp (Get_Flat (Scr, S + 1) - Mx));
                        Den := Den + Get_Flat (Scr, S + 1);
                     end loop;
                     for J in 1 .. HD loop
                        declare Acc : Float := 0.0; begin
                           for S in 0 .. Pos loop
                              Acc := Acc + (Get_Flat (Scr, S + 1) / Den)
                                       * Get_Flat (VC (S + 1), KV * HD + J);
                           end loop;
                           Set_Flat (Ctx_O, Hh * HD + J, Acc);
                        end;
                     end loop;
                  end;
               end loop;
               Add_To (H, LLM_Weight.MatVec (B.W_O, Ctx_O));   -- attn residual
            end;

            --  SwiGLU FFN with residual.
            declare
               Xf   : constant Tensor := GN (H, B.Ffn_Norm);
               Gate : constant Tensor := Silu (LLM_Weight.MatVec (B.W_Gate, Xf));
               Up   : constant Tensor := LLM_Weight.MatVec (B.W_Up, Xf);
            begin
               Add_To (H, LLM_Weight.MatVec (B.W_Down, Gate * Up));
            end;
         end;
      end loop;

      return LLM_Weight.MatVec (M.Output, GN (H, M.Out_Norm));
   end Forward_Step;

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
        (others => 0);
      N_Hist : Natural := 0;

      procedure Free is
        new Ada.Unchecked_Deallocation (Tensor_Array, Tensor_Array_Ptr);
   begin
      for L in 1 .. M.N_Blocks loop
         Cache (L).K := new Tensor_Array (1 .. Cap);
         Cache (L).V := new Tensor_Array (1 .. Cap);
      end loop;

      for I in Ids'Range loop
         Logits := Forward_Step (M, Cache, Ids (I), Len);
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
            Logits := Forward_Step (M, Cache, Tid, Len);
            Len := Len + 1;
         end;
      end loop;

      for L in 1 .. M.N_Blocks loop
         Free (Cache (L).K); Free (Cache (L).V);
      end loop;
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
      return Generate
        (M, One (M.Bos) & Conv_Ids (Conversation'First) & Header ("assistant"),
         -1, -1, Max_New_Tokens, Sink, Params);
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
