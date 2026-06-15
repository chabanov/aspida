---------------------------------------------------------------------
-- LLM_Qwen body — Qwen 3.5 MoE model loader (full implementation)
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Numerics.Elementary_Functions;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Unchecked_Deallocation;
with Ada.Exceptions;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant; use LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;
with LLM_MoE;
with LLM_RMSNorm;
with LLM_FullAttn;
with LLM_DeltaNet_Blk;
with LLM_RoPE;
with LLM_Weight;

package body LLM_Qwen is

   use Ada.Strings.Fixed;

   --  Debug: dump residual-stream norm per layer when LLM_DEBUG_NORMS is set.
   Debug_Norms : constant Boolean :=
     Ada.Environment_Variables.Exists ("LLM_DEBUG_NORMS");

   function "=" (Left, Right : Qwen_Model) return Boolean is
   begin
      return Left.N_Blocks = Right.N_Blocks
        and then Left.Vocab_Sz = Right.Vocab_Sz
        and then Left.Model_Dim = Right.Model_Dim;
   end "=";

   function Img (N : Integer) return String is
   begin
      return Trim (Integer'Image (N), Ada.Strings.Both);
   end Img;

   --------------------------------------------------------------------
   -- Load model from GGUF
   --------------------------------------------------------------------

   function Load (Path : String) return Qwen_Model is
      G    : GGUF_File;
      M    : Qwen_Model;
      Dim  : Integer := 2048;
      N_Layers : Integer := 36;
      N_Experts     : Integer := 256;   -- GGUF override below; Qwen-3.5 default
      Full_Interval : Integer := 4;     -- full-attn every Nth layer (L mod N = N-1)
      RoPE_Dim      : Integer := 64;            -- rope dimension count
      RoPE_Base     : Float   := 10_000_000.0;  -- rope frequency base

      function L (Name : String) return Tensor is
         Info : constant Tensor_Info := Find_Tensor (G, Name);
         Size : constant Natural := Natural (Tensor_Byte_Size (Info));
         type Raw_Access is access String;
         procedure Free is new Ada.Unchecked_Deallocation (String, Raw_Access);
         Raw  : Raw_Access := new String (1 .. Size);
      begin
         Read_Tensor_Raw (G, Info, Raw.all'Address, Size);
         return Result : constant Tensor := Dequantize (Info, Raw.all) do
            Free (Raw);
         end return;
      exception
         when Constraint_Error =>
            raise Model_Load_Error with "tensor not found: " & Name;
         when E : others =>
            raise Model_Load_Error with "error loading tensor " & Name & ": "
              & Ada.Exceptions.Exception_Message (E);
      end L;

      --  Load a weight WITHOUT dequantizing — keeps raw quantized bytes so the
      --  model fits in RAM; projections matvec it on the fly (LLM_Weight).
      function LQ (Name : String) return LLM_Weight.Weight is
         Info : constant Tensor_Info := Find_Tensor (G, Name);
         Size : constant Natural := Natural (Tensor_Byte_Size (Info));
         B    : constant LLM_Weight.Byte_Data := new String (1 .. Size);
      begin
         Read_Tensor_Raw (G, Info, B.all'Address, Size);
         return LLM_Weight.From_Quant (Info, B);
      exception
         when E : others =>
            raise Model_Load_Error with "error loading weight " & Name & ": "
              & Ada.Exceptions.Exception_Message (E);
      end LQ;

   begin
      Ada.Text_IO.Put_Line ("Loading Qwen model from " & Path & " ...");
      Open (G, Path);

      if not Is_Open (G) then
         raise Model_Load_Error with "cannot open GGUF file: " & Path;
      end if;

      --  This engine implements the qwen35moe hybrid (MoE + gated delta-net).
      --  Refuse any other architecture up front with a clear message instead
      --  of crashing later on a missing config key or tensor.
      declare
         Arch : constant String := Metadata (G, "general.architecture");
      begin
         if Arch /= "qwen35moe" and then Arch /= "qwen2" then
            raise Model_Load_Error with
              "unsupported architecture '" & Arch
              & "' — this engine supports qwen35moe only";
         end if;
      end;

      -- Read config from metadata (try qwen35moe.* first, fallback to qwen2.*)
      declare
         function Read_Meta_Int (Key : String) return Integer is
            Val : constant String := Metadata (G, "qwen35moe." & Key);
         begin
            if Val /= "" then
               return Integer'Value (Val);
            end if;
            -- Fallback: try qwen2 prefix
            return Integer'Value (Metadata (G, "qwen2." & Key));
         end Read_Meta_Int;
      begin
         -- Extract key suffix after the architecture prefix
         Dim := Read_Meta_Int ("embedding_length");
         N_Layers := Read_Meta_Int ("block_count");
         M.N_Heads := Read_Meta_Int ("attention.head_count");
         M.Ctx_Len := Read_Meta_Int ("context_length");

         -- vocab_size is not in metadata for qwen35moe — derive it from
         -- the token embedding tensor shape: token_embd.weight = [dim, vocab]
         begin
            M.Vocab_Sz := Read_Meta_Int ("vocab_size");
         exception
            when Constraint_Error =>
               declare
                  TE : constant Tensor_Info := Find_Tensor (G, "token_embd.weight");
               begin
                  -- Dims(1) = embedding_length, Dims(2) = vocab_size
                  if TE.N_Dims >= 2 then
                     M.Vocab_Sz := Integer (TE.Dims (2));
                  else
                     M.Vocab_Sz := Integer (TE.Dims (1));
                  end if;
               end;
         end;

         begin
            M.N_KV_Heads := Read_Meta_Int ("attention.head_count_kv");
         exception
            when others => M.N_KV_Heads := 4;
         end;
      end;

      --  Architecture parameters that were previously hard-coded: read them
      --  from GGUF metadata when present, else keep the known Qwen-3.5 default,
      --  so the bundled model is byte-identical while other configs work.
      declare
         function Meta_Int (Key : String; Default : Integer) return Integer is
            V : constant String := Metadata (G, "qwen35moe." & Key);
         begin
            return (if V = "" then Default else Integer'Value (V));
         exception
            when others => return Default;
         end;
         function Meta_Float (Key : String; Default : Float) return Float is
            V : constant String := Metadata (G, "qwen35moe." & Key);
         begin
            return (if V = "" then Default else Float'Value (V));
         exception
            when others => return Default;
         end;
      begin
         N_Experts := Meta_Int ("expert_count", N_Experts);
         Full_Interval :=
           Meta_Int ("attention.full_attention_interval", Full_Interval);
         if Full_Interval < 1 then
            Full_Interval := 4;   -- guard against a bogus/zero value
         end if;
         RoPE_Dim  := Meta_Int ("rope.dimension_count", RoPE_Dim);
         RoPE_Base := Meta_Float ("rope.freq_base", RoPE_Base);
         if RoPE_Dim < 2 then
            RoPE_Dim := 64;       -- guard against a bogus/zero value
         end if;
      end;
      Ada.Text_IO.Put_Line ("  rope: dim=" & Img (RoPE_Dim)
        & " base=" & Float'Image (RoPE_Base)
        & " experts=" & Img (N_Experts)
        & " full_attn_every=" & Img (Full_Interval));

      -- Allocate block array
      M.Blocks := new Block_Array (1 .. N_Layers);

      -- Persist dimensions on the model record. Without this, M.Model_Dim and
      -- M.N_Blocks stay uninitialized and Forward/Param_Count read garbage.
      M.Model_Dim := Dim;
      M.N_Blocks  := N_Layers;

      Ada.Text_IO.Put_Line ("  dim=" & Img (Dim) & " layers=" & Img (N_Layers) &
        " heads=" & Img (M.N_Heads) & " vocab=" & Img (M.Vocab_Sz));

      -- Token embeddings [vocab, dim] — ~2GB FP32, slow
      Ada.Text_IO.Put_Line ("  loading token_embd (2GB)...");
      Ada.Text_IO.Flush;
      M.Token_Emb := L ("token_embd.weight");
      Ada.Text_IO.Put_Line ("  token_embd loaded.");

      -- Final norm [dim]
      M.Final_Norm := L ("output_norm.weight");

      -- LM head [dim, vocab]  (transposed in Qwen convention) — ~1GB, slow
      Ada.Text_IO.Put_Line ("  loading output.weight (1GB)...");
      Ada.Text_IO.Flush;
      M.LM_Head := L ("output.weight");
      Ada.Text_IO.Put_Line ("  output.weight loaded.");

      Ada.Text_IO.Put_Line ("  Loading transformer blocks...");
      Ada.Text_IO.Flush;

      -- Load all transformer blocks. Layer type: full attention when
      -- L mod 4 = 3 (full_attention_interval), gated delta-net otherwise.
      for I in 0 .. N_Layers - 1 loop
         declare
            Pre     : constant String := "blk." & Img (I) & ".";
            Is_Full : constant Boolean := (I mod Full_Interval) = Full_Interval - 1;
            Blk     : LLM_Qwen_Blk.Qwen_Block;
         begin
            Ada.Text_IO.Put_Line ("  loading block" & Img (I) &
              (if Is_Full then " [full-attn]" else " [delta-net]"));
            Ada.Text_IO.Flush;

            Blk.Is_Full_Attn     := Is_Full;
            Blk.Dim              := Dim;
            Blk.Attn_Norm_W      := L (Pre & "attn_norm.weight");
            Blk.Post_Attn_Norm_W := L (Pre & "post_attention_norm.weight");

            if Is_Full then
               Blk.Full := LLM_FullAttn.Create
                 (LQ (Pre & "attn_q.weight"),
                  LQ (Pre & "attn_k.weight"),
                  LQ (Pre & "attn_v.weight"),
                  L  (Pre & "attn_q_norm.weight"),
                  L  (Pre & "attn_k_norm.weight"),
                  LQ (Pre & "attn_output.weight"),
                  LLM_RoPE.Create_Qwen_RoPE (RoPE_Dim, RoPE_Base, M.Ctx_Len));
            else
               Blk.DNet := LLM_DeltaNet_Blk.Create
                 (LQ (Pre & "attn_qkv.weight"),
                  L  (Pre & "ssm_conv1d.weight"),
                  L  (Pre & "ssm_a"),
                  L  (Pre & "ssm_dt.bias"),
                  LQ (Pre & "ssm_alpha.weight"),
                  LQ (Pre & "ssm_beta.weight"),
                  L  (Pre & "ssm_norm.weight"),
                  LQ (Pre & "ssm_out.weight"),
                  LQ (Pre & "attn_gate.weight"));
            end if;

            --  MoE FFN on every block.
            Blk.MoE := LLM_MoE.Create_MoE
              (LQ (Pre & "ffn_gate_inp.weight"),
               LQ (Pre & "ffn_gate_exps.weight"),
               LQ (Pre & "ffn_up_exps.weight"),
               LQ (Pre & "ffn_down_exps.weight"),
               LQ (Pre & "ffn_gate_shexp.weight"),
               LQ (Pre & "ffn_up_shexp.weight"),
               LQ (Pre & "ffn_down_shexp.weight"),
               L  (Pre & "ffn_gate_inp_shexp.weight"),
               N_Experts);

            M.Blocks (I + 1) := new LLM_Qwen_Blk.Qwen_Block'(Blk);
         end;
      end loop;

      -- Build the tokenizer from the GGUF vocab/merges (byte-level fallback
      -- if the file has no tokenizer arrays).
      M.Tok := LLM_Tokenizer.Create;
      LLM_Tokenizer.Load_From_GGUF (M.Tok, G);
      Ada.Text_IO.Put_Line ("  tokenizer: " &
        Img (LLM_Tokenizer.Vocab_Size (M.Tok)) & " tokens.");

      --  Resolve ChatML control tokens for the chat layer (-1 if absent).
      M.Im_Start_Id := LLM_Tokenizer.Token_To_Id (M.Tok, "<|im_start|>");
      M.Im_End_Id   := LLM_Tokenizer.Token_To_Id (M.Tok, "<|im_end|>");
      begin
         M.Eos_Id := Integer'Value (Metadata (G, "tokenizer.ggml.eos_token_id"));
      exception
         when others => M.Eos_Id := M.Im_End_Id;
      end;
      Ada.Text_IO.Put_Line ("  chat tokens: im_start=" & Img (M.Im_Start_Id)
        & " im_end=" & Img (M.Im_End_Id) & " eos=" & Img (M.Eos_Id)
        & " unk=" & Img (LLM_Tokenizer.Unk_Id (M.Tok)));

      Close (G);
      Ada.Text_IO.Put_Line ("  Qwen model loaded: " & Img (M.N_Blocks) & " blocks.");
      return M;
   end Load;

   --------------------------------------------------------------------
   -- Forward pass: token_ids [seq_len] → next-token logits [1, vocab]
   --------------------------------------------------------------------

   function Forward (M : Qwen_Model; Token_Ids : Tensor) return Tensor is
      Dim     : constant Integer := M.Model_Dim;
      Seq_Len : constant Integer := Numel (Token_Ids);
      H       : Tensor;
   begin
      if Seq_Len < 1 then
         return New_Tensor ([1, M.Vocab_Sz]);
      end if;

      -- Build the embedding sequence [Seq_Len, Dim]: one row per token.
      H := New_Tensor ([Seq_Len, Dim]);
      for Pos in 1 .. Seq_Len loop
         declare
            Tid : Integer := Integer (Get_Flat (Token_Ids, Pos));
         begin
            if Tid < 1 then
               Tid := 1;
            elsif Tid > M.Vocab_Sz then
               Tid := M.Vocab_Sz;
            end if;
            for D in 1 .. Dim loop
               Set (H, [Pos, D], Get (M.Token_Emb, [Tid, D]));
            end loop;
         end;
      end loop;

      -- Run the transformer blocks over the whole sequence.
      for I in 1 .. M.N_Blocks loop
         H := LLM_Qwen_Blk.Forward (M.Blocks (I).all, H);
         if Debug_Norms then
            declare
               SS : Float := 0.0;
            begin
               for D in 1 .. Dim loop
                  SS := SS + Get (H, [Seq_Len, D]) ** 2;
               end loop;
               Ada.Text_IO.Put_Line ("  [L" & Img (I) & "] "
                 & (if M.Blocks (I).all.Is_Full_Attn then "full " else "dnet ")
                 & "|H|=" & Float'Image (Ada.Numerics.Elementary_Functions.Sqrt (SS))
                 & "  h0..3=" & Float'Image (Get (H, [Seq_Len, 1]))
                 & Float'Image (Get (H, [Seq_Len, 2]))
                 & Float'Image (Get (H, [Seq_Len, 3]))
                 & Float'Image (Get (H, [Seq_Len, 4])));
            end;
         end if;
      end loop;

      -- Final RMSNorm on the last position, then project to vocab logits.
      declare
         Last : Tensor := New_Tensor ([1, Dim]);
      begin
         for D in 1 .. Dim loop
            Set_Flat (Last, D, Get (H, [Seq_Len, D]));
         end loop;
         declare
            Normed : constant Tensor := LLM_RMSNorm.Forward (Last, M.Final_Norm);
         begin
            --  output.weight is [vocab, dim] (row-major); flat parallel matvec.
            return MatVec_Rows (M.LM_Head, Normed);
         end;
      end;
   end Forward;

   --------------------------------------------------------------------
   -- Generate (autoregressive loop)
   --------------------------------------------------------------------

   --  Shared cached-decode core: prefill the prompt token ids, then greedily
   --  generate up to Max_New_Tokens, stopping early at Stop1/Stop2 (ids; -1 =
   --  none). Returns the decoded text of the GENERATED tokens only.
   function Decode_Tokens
     (M              : Qwen_Model;
      Prompt_Ids     : LLM_Tokenizer.Token_Array;
      Max_New_Tokens : Integer;
      Stop1, Stop2   : Integer;
      Sink           : access Token_Sink'Class;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy) return String
   is
      Dim     : constant Integer := M.Model_Dim;
      Cap     : constant Integer :=
        Integer'Max (1, Prompt_Ids'Length + Max_New_Tokens);
      Out_Buf : Unbounded_String;
      Smp     : LLM_Sampler.Sampler := LLM_Sampler.Create (Params);
      Hist    : LLM_Sampler.History (1 .. Integer'Max (1, Max_New_Tokens)) :=
        (others => 0);
      N_Hist  : Natural := 0;

      --  Per-layer decode state (KV cache for full-attn, recurrent state +
      --  conv window for delta-net), threaded across tokens. One forward
      --  step costs O(1) matmuls instead of recomputing the sequence.
      Cache : array (1 .. M.N_Blocks) of LLM_Qwen_Blk.Block_State;

      function Decode (Embed_Row : Integer) return Tensor is
         H : Tensor := New_Tensor ([1, Dim]);
      begin
         for D in 1 .. Dim loop
            Set_Flat (H, D, Get (M.Token_Emb, [Embed_Row, D]));
         end loop;
         for I in 1 .. M.N_Blocks loop
            H := LLM_Qwen_Blk.Step (M.Blocks (I).all, Cache (I), H);
         end loop;
         declare
            Normed : constant Tensor := LLM_RMSNorm.Forward (H, M.Final_Norm);
         begin
            return MatVec_Rows (M.LM_Head, Normed);
         end;
      end Decode;

      Last_Logits : Tensor;
   begin
      for I in 1 .. M.N_Blocks loop
         Cache (I) := LLM_Qwen_Blk.Init_State (M.Blocks (I).all, Cap);
      end loop;

      --  Prefill (row = id + 1: ids are 0-based, embedding rows 1-based).
      --  Tick per token so a UI shows progress before the first output token.
      if Prompt_Ids'Length = 0 then
         Last_Logits := Decode (1);
      else
         for I in Prompt_Ids'Range loop
            Last_Logits := Decode (Prompt_Ids (I) + 1);
            if Sink /= null then
               Sink.Tick;
            end if;
         end loop;
      end if;

      for Step in 1 .. Max_New_Tokens loop
         declare
            Win : constant Natural :=
              Integer'Min (N_Hist, Integer'Max (0, Params.Repeat_Last_N));
            Tid : constant Integer := LLM_Sampler.Next
              (Smp, Last_Logits, Hist (N_Hist - Win + 1 .. N_Hist));
            Best_Row : constant Integer := Tid + 1;   -- 1-based embedding row
         begin
            exit when Best_Row < 1 or else Best_Row > M.Vocab_Sz;
            exit when Tid = Stop1 or else Tid = Stop2;     -- stop tokens
            declare
               Piece : constant String := LLM_Tokenizer.Decode_One (M.Tok, Tid);
            begin
               Append (Out_Buf, Piece);
               if Sink /= null then            -- stream this token now
                  Sink.Emit (Piece);
               end if;
            end;
            N_Hist := N_Hist + 1; Hist (N_Hist) := Tid;
            exit when Step = Max_New_Tokens;
            Last_Logits := Decode (Best_Row);
         end;
      end loop;

      return To_String (Out_Buf);
   end Decode_Tokens;

   function Generate
     (M : Qwen_Model; Prompt : String; Max_New_Tokens : Integer := 128;
      Sink : access Token_Sink'Class := null) return String
   is
      Ids : constant LLM_Tokenizer.Token_Array := LLM_Tokenizer.Encode (M.Tok, Prompt);
   begin
      --  Raw completion: no chat template, no stop token (legacy behaviour).
      return Prompt & Decode_Tokens (M, Ids, Max_New_Tokens, -1, -1, Sink);
   end Generate;

   --  Drop a leading <think>...</think> reasoning block and any whitespace
   --  before the answer; if the block never closed (ran past the budget),
   --  keep the raw text so nothing is silently lost.
   function Strip_Think (S : String) return String is
      Close_Tag : constant String := "</think>";
      Idx       : constant Natural := Index (S, Close_Tag);
      First     : Integer;
   begin
      if Idx = 0 then
         return S;
      end if;
      First := Idx + Close_Tag'Length;
      while First <= S'Last
        and then (S (First) = ' ' or else S (First) = ASCII.LF
                  or else S (First) = ASCII.CR or else S (First) = ASCII.HT)
      loop
         First := First + 1;
      end loop;
      return S (First .. S'Last);
   end Strip_Think;

   function One (Id : Integer) return LLM_Tokenizer.Token_Array is
     [1 => Id];

   --  A control marker resolved to its single special-token id if the vocab
   --  has one, else its byte-BPE pieces — works whether <think> etc. are
   --  special tokens or plain text.
   function Marker (M : Qwen_Model; Piece : String)
      return LLM_Tokenizer.Token_Array
   is
      Id : constant Integer := LLM_Tokenizer.Token_To_Id (M.Tok, Piece);
   begin
      if Id >= 0 then
         return One (Id);
      else
         return LLM_Tokenizer.Encode (M.Tok, Piece);
      end if;
   end Marker;

   function Role_Str (R : Role_Kind) return String is
     (case R is when Role_System => "system", when Role_User => "user",
         when Role_Assistant => "assistant");

   --  Token ids for one message: <|im_start|>{role}\n{text}<|im_end|>\n
   function Msg_Ids (M : Qwen_Model; Msg : Message)
      return LLM_Tokenizer.Token_Array
   is
      LF : constant String := [1 => ASCII.LF];
      use type LLM_Tokenizer.Token_Array;
   begin
      return One (M.Im_Start_Id)
        & LLM_Tokenizer.Encode (M.Tok, Role_Str (Msg.Role) & LF)
        & LLM_Tokenizer.Encode (M.Tok, To_String (Msg.Text))
        & One (M.Im_End_Id)
        & LLM_Tokenizer.Encode (M.Tok, LF);
   end Msg_Ids;

   --  Token ids for messages First .. Last (recursive concat; avoids growing
   --  a constrained array).
   function Conv_Ids
     (M : Qwen_Model; Conv : Message_Array; I : Positive)
      return LLM_Tokenizer.Token_Array
   is
      use type LLM_Tokenizer.Token_Array;
   begin
      if I >= Conv'Last then
         return Msg_Ids (M, Conv (I));
      else
         return Msg_Ids (M, Conv (I)) & Conv_Ids (M, Conv, I + 1);
      end if;
   end Conv_Ids;

   function Chat
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String
   is
      LF : constant String := [1 => ASCII.LF];
      use type LLM_Tokenizer.Token_Array;
   begin
      --  Fall back to raw generation if the model has no ChatML tokens.
      if M.Im_Start_Id < 0 or else M.Im_End_Id < 0 then
         return Generate
           (M, To_String (Conversation (Conversation'Last).Text),
            Max_New_Tokens, Sink);
      end if;

      --  Full transcript, then the assistant turn with thinking disabled
      --  (empty <think></think> — Qwen3 enable_thinking=False convention).
      declare
         Ids : constant LLM_Tokenizer.Token_Array :=
           Conv_Ids (M, Conversation, Conversation'First)
           & One (M.Im_Start_Id)
           & LLM_Tokenizer.Encode (M.Tok, "assistant" & LF)
           & Marker (M, "<think>")
           & LLM_Tokenizer.Encode (M.Tok, LF & LF)
           & Marker (M, "</think>")
           & LLM_Tokenizer.Encode (M.Tok, LF & LF);
         Raw : constant String :=
           Decode_Tokens (M, Ids, Max_New_Tokens, M.Im_End_Id, M.Eos_Id, Sink, Params);
      begin
         return Strip_Think (Raw);
      end;
   end Chat;

   function Chat
     (M : Qwen_Model; User : String; Max_New_Tokens : Integer := 256;
      Sink : access Token_Sink'Class := null) return String
   is
   begin
      return Chat
        (M, Message_Array'(1 => (Role_User, To_Unbounded_String (User))),
         Max_New_Tokens, Sink);
   end Chat;

   -- Parameter count (total FP32 params after dequantization)
   -- Safe addition: only add if tensor has elements (skip uninit tensors)
   function Safe_N (T : Tensor) return Long_Long_Integer is
   begin
      if Numel (T) > 0 then
         return Long_Long_Integer (Numel (T));
      else
         return 0;
      end if;
   end Safe_N;

   function Param_Count (M : Qwen_Model) return Long_Long_Integer is
      C : Long_Long_Integer := 0;
   begin
      C := Safe_N (M.Token_Emb) + Safe_N (M.LM_Head) + Safe_N (M.Final_Norm);
      for I in 1 .. M.N_Blocks loop
         declare
            B : LLM_Qwen_Blk.Qwen_Block renames M.Blocks (I).all;
         begin
            C := C + Safe_N (B.Attn_Norm_W) + Safe_N (B.Post_Attn_Norm_W);
            if B.Is_Full_Attn then
               C := C + LLM_Weight.Count (B.Full.Q_W) + LLM_Weight.Count (B.Full.K_W)
                      + LLM_Weight.Count (B.Full.V_W) + Safe_N (B.Full.Q_Norm)
                      + Safe_N (B.Full.K_Norm) + LLM_Weight.Count (B.Full.O_W);
            else
               C := C + LLM_Weight.Count (B.DNet.QKV_W) + Safe_N (B.DNet.Conv_W)
                      + Safe_N (B.DNet.A_W) + Safe_N (B.DNet.Dt_W)
                      + LLM_Weight.Count (B.DNet.Alpha_W) + LLM_Weight.Count (B.DNet.Beta_W)
                      + Safe_N (B.DNet.Norm_W) + LLM_Weight.Count (B.DNet.Out_W)
                      + LLM_Weight.Count (B.DNet.Gate_W);
            end if;
            C := C + LLM_Weight.Count (B.MoE.Gate_Inp_W)
                    + LLM_Weight.Count (B.MoE.Gate_Exp_W)
                    + LLM_Weight.Count (B.MoE.Up_W)
                    + LLM_Weight.Count (B.MoE.Down_W)
                    + LLM_Weight.Count (B.MoE.Shexp_Gate_W)
                    + LLM_Weight.Count (B.MoE.Shexp_Up_W)
                    + LLM_Weight.Count (B.MoE.Shexp_Down_W)
                    + Safe_N (B.MoE.Shexp_Gate_Inp_W);
         end;
      end loop;
      return C;
   end Param_Count;

   function Vocab_Size  (M : Qwen_Model) return Integer is (M.Vocab_Sz);
   function Context_Len (M : Qwen_Model) return Integer is (M.Ctx_Len);
   function Dim         (M : Qwen_Model) return Integer is (M.Model_Dim);
   function Block_Count (M : Qwen_Model) return Integer is (M.N_Blocks);

end LLM_Qwen;
