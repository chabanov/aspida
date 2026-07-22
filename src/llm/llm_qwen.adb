---------------------------------------------------------------------
-- LLM_Qwen body — Qwen 3.5 MoE model loader (full implementation)
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Real_Time;
with Ada.Numerics.Elementary_Functions;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Unchecked_Deallocation;
with Ada.Exceptions;
with Ada.Finalization;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant; use LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;
with LLM_MoE;
with LLM_Dense_FFN;
with LLM_GPU;
with LLM_Qwen_GPU;
with System;
with Interfaces.C;
with LLM_Step_Lock;
with LLM_Batcher;
with LLM_RMSNorm;
with LLM_FullAttn;
with LLM_DeltaNet_Blk;
with LLM_RoPE;
with LLM_Chat_Parser;

package body LLM_Qwen is

   use Ada.Strings.Fixed;

   --  Decode profiler (env ASPIDA_PROFILE): block-loop vs LM-head per token.
   Prof2_On   : constant Boolean := Ada.Environment_Variables.Exists ("ASPIDA_PROFILE");
   Prof2_Loop : Duration := 0.0;
   Prof2_Head : Duration := 0.0;
   Prof2_N    : Natural  := 0;

   procedure Prof2_Tick is
   begin
      Prof2_N := Prof2_N + 1;
      if Prof2_N >= 100 then
         Ada.Text_IO.Put_Line
           (Ada.Text_IO.Standard_Error,
            "[PROF drv] per-token: blocks=" & Duration'Image (Prof2_Loop / 100.0)
            & "s lm_head=" & Duration'Image (Prof2_Head / 100.0) & "s");
         Prof2_Loop := 0.0; Prof2_Head := 0.0; Prof2_N := 0;
      end if;
   end Prof2_Tick;

   --  Default Chat_Sink.Emit / Tick forward to On_Text so a sink that
   --  overrides only On_Text still receives text pieces (legacy Token_Sink
   --  compatibility). The chat layer routes reasoning/tools via their
   --  dedicated callbacks, never Emit.
   overriding procedure Emit (S : in out Chat_Sink; Piece : String) is
   begin
      On_Text (S, Piece);
   end Emit;

   overriding procedure Tick (S : in out Chat_Sink) is
   begin
      null;
   end Tick;

   function New_Null_Sink return Null_Sink_Access is
      Result : constant Null_Sink_Access := new Null_Sink;
   begin
      return Result;
   end New_Null_Sink;

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
      N_Experts     : Integer := 256;   -- GGUF override below; Qwen-3.5 MoE default
      Is_Dense      : Boolean := False;  -- no expert_count key => dense SwiGLU FFN
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
         if Arch /= "qwen35moe" and then Arch /= "qwen35"
           and then Arch /= "qwen2"
         then
            raise Model_Load_Error with
              "unsupported architecture '" & Arch
              & "' — this backend supports qwen35moe, qwen35 and qwen2";
         end if;
         M.Arch := To_Unbounded_String (Arch);
      end;

      -- Read config from metadata (try qwen35moe.* first, fallback to qwen2.*)
      declare
         function Read_Meta_Int (Key : String) return Integer is
            V_Moe : constant String := Metadata (G, "qwen35moe." & Key);
            V_Q35 : constant String := Metadata (G, "qwen35." & Key);
         begin
            -- Prefix chain: qwen35moe. -> qwen35. -> qwen2.
            if V_Moe /= "" then
               return Integer'Value (V_Moe);
            elsif V_Q35 /= "" then
               return Integer'Value (V_Q35);
            end if;
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
         --  Look up Key under qwen35moe. then qwen35. (empty if neither).
         function Meta (Key : String) return String is
            V_Moe : constant String := Metadata (G, "qwen35moe." & Key);
         begin
            return (if V_Moe /= "" then V_Moe
                    else Metadata (G, "qwen35." & Key));
         end Meta;
         function Meta_Int (Key : String; Default : Integer) return Integer is
            V : constant String := Meta (Key);
         begin
            return (if V = "" then Default else Integer'Value (V));
         exception
            when others => return Default;
         end;
         function Meta_Float (Key : String; Default : Float) return Float is
            V : constant String := Meta (Key);
         begin
            return (if V = "" then Default else Float'Value (V));
         exception
            when others => return Default;
         end;
      begin
         --  Dense vs MoE is decided by the presence of expert_count: absent
         --  (sentinel -1) => dense SwiGLU FFN (qwen35); present and > 0 => MoE.
         N_Experts := Meta_Int ("expert_count", -1);
         Is_Dense  := N_Experts <= 0;
         if Is_Dense then
            N_Experts := 0;
         end if;
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
        & (if Is_Dense then " ffn=dense"
           else " experts=" & Img (N_Experts))
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
      M.LM_Head_Q := LQ ("output.weight");   -- native quant for the GPU chain
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

            --  FFN on every block: dense SwiGLU (qwen35) or routed MoE
            --  (qwen35moe). Decided once at load from expert_count presence.
            Blk.Is_MoE := not Is_Dense;
            if Is_Dense then
               Blk.Dense := LLM_Dense_FFN.Create
                 (LQ (Pre & "ffn_gate.weight"),
                  LQ (Pre & "ffn_up.weight"),
                  LQ (Pre & "ffn_down.weight"));
            else
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
            end if;

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
   --  Free — full teardown for Phase 1b LRU eviction.
   --
   --  Qwen_Model is a by-value record (not an access type): the record itself
   --  and its dense controlled Tensors (Token_Emb / LM_Head / Final_Norm) are
   --  finalized when the enclosing backend object is deallocated. Free only
   --  releases what the record OWNS on the heap and would otherwise leak: the
   --  per-block weight bytes (+ any GPU mirror) and the block array/records.
   --------------------------------------------------------------------
   procedure Free_Block is
     new Ada.Unchecked_Deallocation (LLM_Qwen_Blk.Qwen_Block, Block_Access);
   procedure Free_Block_Arr is
     new Ada.Unchecked_Deallocation (Block_Array, Block_Array_Ptr);

   --  Phase C: full resident forward chain — registered once per loaded model,
   --  keyed by the embedding table address. Package-level because the resident
   --  GPU state it tracks (g_chain in the shim) is itself one-per-process.
   --  Declared HERE, ahead of Free, so Free can reset it on eviction; that is
   --  load-bearing, see the comment in Free.
   Chain_Tag  : System.Address := System.Null_Address;
   Chain_OK   : Boolean := False;
   Bench_Done : Boolean := False;

   procedure Free (M : in out Qwen_Model) is
   begin
      if M.Blocks /= null then
         for I in M.Blocks'Range loop
            if M.Blocks (I) /= null then
               --  Free both attention paths' weights: the unused one carries
               --  no bytes (Free_Bytes is idempotent on an empty weight).
               LLM_FullAttn.Free (M.Blocks (I).Full);
               LLM_DeltaNet_Blk.Free (M.Blocks (I).DNet);
               --  Free whichever FFN this block carries (the other holds no
               --  bytes; Free is idempotent on an empty layer).
               LLM_MoE.Free (M.Blocks (I).MoE);
               LLM_Dense_FFN.Free (M.Blocks (I).Dense);
               --  Attn_Norm_W / Post_Attn_Norm_W are controlled Tensors,
               --  finalized when the block record is deallocated next.
               Free_Block (M.Blocks (I));
            end if;
         end loop;
         Free_Block_Arr (M.Blocks);   --  nulls M.Blocks -> idempotent
      end if;
      --  Model-level weights. Token_Emb / Final_Norm are controlled Tensors and
      --  finalize themselves, but LM_Head_Q is an LLM_Weight.Weight -- a plain
      --  record owning heap bytes plus a GPU mirror keyed on their address, so
      --  nothing reclaims it implicitly. Llama and Gemma already drop their
      --  model-level weights this way; Qwen leaked output.weight (~0.5 GB at
      --  35B Q8_0) on every eviction. Both calls are idempotent on an empty
      --  weight, so a second Free is a no-op.
      LLM_GPU.Free_Weight (LLM_Weight.Raw_Address (M.LM_Head_Q));
      LLM_Weight.Free_Bytes (M.LM_Head_Q);

      --  Reset the resident-chain registration. Chain_Tag caches the embedding
      --  table's ADDRESS as an identity for "this model's chain is already on
      --  the GPU". Without clearing it here, evicting this model leaves the tag
      --  pointing at freed memory: the allocator can hand the same address to
      --  the NEXT model's token_embd (it is the first and largest allocation of
      --  a load), Register_Chain's `Chain_Tag = Data_Address (M.Token_Emb)`
      --  early-out then fires, the new model is NOT re-registered, and decode
      --  runs it through the previous model's resident device weights — silent
      --  wrong output, no error. Chain_Reset also clears the shim's g_chain so
      --  the stale device pointers cannot be reused. Only reached under
      --  multi-model eviction (ASPIDA_MAX_LOADED_MODELS with a non-default
      --  slot); a single pinned model never frees.
      Chain_Tag  := System.Null_Address;
      Chain_OK   := False;
      Bench_Done := False;
      LLM_Qwen_GPU.Chain_Reset;
   end Free;

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
   --  Forward_Logits — like Forward, but the final RMSNorm + head are applied
   --  at EVERY position (not just the last), yielding per-position logits for
   --  distillation. Embedding rows follow the generation path (row = id + 1).
   --------------------------------------------------------------------
   function Forward_Logits
     (M : Qwen_Model; Ids : LLM_Tokenizer.Token_Array) return Logits_Flat
   is
      Dim : constant Integer := M.Model_Dim;
      Vc  : constant Integer := M.Vocab_Sz;
      N   : constant Integer := Ids'Length;
      H   : Tensor;
   begin
      --  Extended return: the [N*Vocab] result is built on the (heap-backed)
      --  secondary stack, so a large vocab never lands a megabyte array on the
      --  primary stack.
      return Res : Logits_Flat (0 .. Integer'Max (0, N * Vc - 1)) :=
        [others => 0.0]
      do
         if N >= 1 then
            --  Build the embedding sequence [N, Dim].
            H := New_Tensor ([N, Dim]);
            for Pos in 1 .. N loop
               declare
                  Row : Integer := Ids (Ids'First + Pos - 1) + 1;  -- id -> row
               begin
                  if Row < 1 then
                     Row := 1;
                  elsif Row > Vc then
                     Row := Vc;
                  end if;
                  for D in 1 .. Dim loop
                     Set (H, [Pos, D], Get (M.Token_Emb, [Row, D]));
                  end loop;
               end;
            end loop;

            for I in 1 .. M.N_Blocks loop
               H := LLM_Qwen_Blk.Forward (M.Blocks (I).all, H);
            end loop;

            --  Final RMSNorm + head at every position.
            for Pos in 1 .. N loop
               declare
                  Last : Tensor := New_Tensor ([1, Dim]);
               begin
                  for D in 1 .. Dim loop
                     Set_Flat (Last, D, Get (H, [Pos, D]));
                  end loop;
                  declare
                     Normed : constant Tensor :=
                       LLM_RMSNorm.Forward (Last, M.Final_Norm);
                     RL     : constant Tensor := MatVec_Rows (M.LM_Head, Normed);
                  begin
                     for K in 1 .. Vc loop
                        Res ((Pos - 1) * Vc + (K - 1)) := Get_Flat (RL, K);
                     end loop;
                  end;
               end;
            end loop;
         end if;
      end return;
   end Forward_Logits;

   --------------------------------------------------------------------
   -- Generate (autoregressive loop)
   --------------------------------------------------------------------

   --  Shared cached-decode core: prefill the prompt token ids, then greedily
   --  generate up to Max_New_Tokens, stopping early at Stop1/Stop2 (ids; -1 =
   --  none). Returns the decoded text of the GENERATED tokens only.
   procedure Register_Chain (M : Qwen_Model) is
      use type System.Address;
      use LLM_Qwen_GPU;

      function WOK (W : LLM_Weight.Weight) return Boolean is
        (LLM_Weight.Kind_Code (W) >= 0 or else LLM_Weight.Is_F32 (W));

      function GW (W : LLM_Weight.Weight) return GPU_Weight is
        (Addr  => LLM_Weight.Raw_Address (W),
         Bytes => LLM_Weight.Raw_Bytes (W),
         Kind  => LLM_Weight.Kind_Code (W));

      function TB (T : Tensor) return Long_Long_Integer is
        (Long_Long_Integer (Numel (T)) * 4);

      function Sec_Total (R : LLM_RoPE.RoPE_Params) return Integer is
         T : Integer := 0;
      begin
         for S in 1 .. 4 loop
            T := T + Integer'Max (0, Integer (Get_Flat (R.Sections, S)));
         end loop;
         return T;
      end Sec_Total;

      OK : Boolean := True;
   begin
      if not Chain_Available then
         return;
      end if;
      if Chain_Tag = Data_Address (M.Token_Emb) then
         return;                       -- this model is already registered
      end if;
      Chain_OK  := False;
      Chain_Tag := System.Null_Address;

      --  Eligibility: every layer must be MoE with K-quant experts and
      --  K-quant|F32 projections (same gates as the per-layer paths).
      for I in 1 .. M.N_Blocks loop
         declare
            B : LLM_Qwen_Blk.Qwen_Block renames M.Blocks (I).all;
         begin
            OK := OK and then B.Is_MoE;
            if B.Is_Full_Attn then
               OK := OK and then WOK (B.Full.Q_W) and then WOK (B.Full.K_W)
                 and then WOK (B.Full.V_W) and then WOK (B.Full.O_W);
            else
               OK := OK and then WOK (B.DNet.QKV_W) and then WOK (B.DNet.Alpha_W)
                 and then WOK (B.DNet.Beta_W) and then WOK (B.DNet.Gate_W)
                 and then WOK (B.DNet.Out_W);
            end if;
            if B.Is_MoE then
               OK := OK and then WOK (B.MoE.Gate_Inp_W)
                 and then LLM_Weight.Kind_Code (B.MoE.Gate_Exp_W) >= 0
                 and then LLM_Weight.Kind_Code (B.MoE.Up_W) >= 0
                 and then LLM_Weight.Kind_Code (B.MoE.Down_W) >= 0
                 and then LLM_Weight.Kind_Code (B.MoE.Shexp_Gate_W) >= 0
                 and then LLM_Weight.Kind_Code (B.MoE.Shexp_Up_W) >= 0
                 and then LLM_Weight.Kind_Code (B.MoE.Shexp_Down_W) >= 0;
            end if;
         end;
      end loop;
      if not OK then
         return;
      end if;

      Chain_Reset;
      for I in 1 .. M.N_Blocks loop
         declare
            B : LLM_Qwen_Blk.Qwen_Block renames M.Blocks (I).all;
         begin
            if B.Is_Full_Attn then
               Chain_Fattn
                 (Attn_Norm => Data_Address (B.Attn_Norm_W),
                  AN_B      => TB (B.Attn_Norm_W),
                  Post_Norm => Data_Address (B.Post_Attn_Norm_W),
                  PN_B      => TB (B.Post_Attn_Norm_W),
                  Q_W => GW (B.Full.Q_W), K_W => GW (B.Full.K_W),
                  V_W => GW (B.Full.V_W), O_W => GW (B.Full.O_W),
                  Q_Norm => Data_Address (B.Full.Q_Norm),
                  QN_B   => TB (B.Full.Q_Norm),
                  K_Norm => Data_Address (B.Full.K_Norm),
                  KN_B   => TB (B.Full.K_Norm),
                  NQ  => B.Full.N_Q_Heads,
                  NKV => B.Full.N_KV_Heads,
                  HD  => B.Full.Head_Dim,
                  RD  => B.Full.RoPE.Dim,
                  Base => B.Full.RoPE.Freq_Base,
                  Freq_Scale => B.Full.RoPE.Freq_Scale,
                  M_Scale => B.Full.RoPE.M_Scale,
                  Yarn_On => (if B.Full.RoPE.Yarn_On then 1 else 0),
                  Corr_Lo => B.Full.RoPE.Corr_Low,
                  Corr_Hi => B.Full.RoPE.Corr_High,
                  FF => (if B.Full.RoPE.Use_FF
                         then Data_Address (B.Full.RoPE.Freq_Factors)
                         else System.Null_Address),
                  FF_B => (if B.Full.RoPE.Use_FF
                           then TB (B.Full.RoPE.Freq_Factors) else 0),
                  Use_FF => (if B.Full.RoPE.Use_FF then 1 else 0),
                  Interleaved => (if B.Full.RoPE.Interleaved then 1 else 0),
                  Sec_Total => Sec_Total (B.Full.RoPE));
            else
               Chain_Dnet
                 (Attn_Norm => Data_Address (B.Attn_Norm_W),
                  AN_B      => TB (B.Attn_Norm_W),
                  Post_Norm => Data_Address (B.Post_Attn_Norm_W),
                  PN_B      => TB (B.Post_Attn_Norm_W),
                  QKV_W => GW (B.DNet.QKV_W), Alpha_W => GW (B.DNet.Alpha_W),
                  Beta_W => GW (B.DNet.Beta_W), Gate_W => GW (B.DNet.Gate_W),
                  Out_W => GW (B.DNet.Out_W),
                  Conv_W => Data_Address (B.DNet.Conv_W),
                  Conv_B => TB (B.DNet.Conv_W),
                  A_W => Data_Address (B.DNet.A_W), A_B => TB (B.DNet.A_W),
                  Dt_W => Data_Address (B.DNet.Dt_W), Dt_B => TB (B.DNet.Dt_W),
                  Norm_W => Data_Address (B.DNet.Norm_W),
                  Norm_B => TB (B.DNet.Norm_W),
                  NV => B.DNet.N_V_Heads,
                  KHD => B.DNet.Key_Head_Dim,
                  VHD => B.DNet.Value_Head_Dim,
                  QO => B.DNet.QKV_Out,
                  Q_Dim => B.DNet.N_K_Heads * B.DNet.Key_Head_Dim,
                  N_K_Heads => B.DNet.N_K_Heads,
                  V_Dim => B.DNet.V_Dim,
                  Kernel => Shape (B.DNet.Conv_W) (2));
            end if;
            Chain_MoE
              (Router => GW (B.MoE.Gate_Inp_W),
               Gate_Exp => GW (B.MoE.Gate_Exp_W),
               Up_Exp => GW (B.MoE.Up_W),
               Down_Exp => GW (B.MoE.Down_W),
               Shared_Gate => GW (B.MoE.Shexp_Gate_W),
               Shared_Up => GW (B.MoE.Shexp_Up_W),
               Shared_Down => GW (B.MoE.Shexp_Down_W),
               SGI => (if Numel (B.MoE.Shexp_Gate_Inp_W) > 1
                       then Data_Address (B.MoE.Shexp_Gate_Inp_W)
                       else System.Null_Address),
               SGI_B => TB (B.MoE.Shexp_Gate_Inp_W),
               SGI_Len => Numel (B.MoE.Shexp_Gate_Inp_W),
               N_Experts => B.MoE.N_Experts,
               Top_K => B.MoE.Top_K,
               Intermed => B.MoE.Intermed);
         end;
      end loop;
      declare
         LM_Is_Q : constant Boolean := LLM_Weight.Kind_Code (M.LM_Head_Q) >= 0;
      begin
         Chain_Model
           (Embed => Data_Address (M.Token_Emb), Embed_B => TB (M.Token_Emb),
            FNorm => Data_Address (M.Final_Norm), FNorm_B => TB (M.Final_Norm),
            LM =>
              (if LM_Is_Q then LLM_Weight.Raw_Address (M.LM_Head_Q)
               else Data_Address (M.LM_Head)),
            LM_B =>
              (if LM_Is_Q then LLM_Weight.Raw_Bytes (M.LM_Head_Q)
               else TB (M.LM_Head)),
            LM_K => (if LM_Is_Q then LLM_Weight.Kind_Code (M.LM_Head_Q) else -1),
            Dim => M.Model_Dim, Vocab => M.Vocab_Sz);
      end;
      Chain_OK  := Chain_Ready;
      Chain_Tag := Data_Address (M.Token_Emb);
      --  Phase E: batched-throughput benchmark (env ASPIDA_BENCH_BATCH=B).
      if Chain_OK and then not Bench_Done
        and then Ada.Environment_Variables.Exists ("ASPIDA_BENCH_BATCH")
        and then LLM_Qwen_GPU.Chain_Batch_Available
      then
         Bench_Done := True;
         declare
            use type Ada.Real_Time.Time;
            B  : constant Integer :=
              Integer'Value (Ada.Environment_Variables.Value ("ASPIDA_BENCH_BATCH"));
            NL : constant Integer := M.N_Blocks;
            NT : constant Integer := 100;
            Cap : constant Integer := NT + 8;
            Handles : array (0 .. B * NL - 1) of Interfaces.C.int := [others => 0];
            Rows : array (0 .. B - 1) of Interfaces.C.int := [others => 1];
            Pos  : array (0 .. B - 1) of Interfaces.C.int := [others => 0];
            Logits : constant Tensor := New_Tensor ([1, B * M.Vocab_Sz]);
            TS : Ada.Real_Time.Time;
         begin
            for Bi in 0 .. B - 1 loop
               for Li in 1 .. NL loop
                  declare
                     Blk : LLM_Qwen_Blk.Qwen_Block renames M.Blocks (Li).all;
                     Hn  : Integer;
                  begin
                     if Blk.Is_Full_Attn then
                        Hn := LLM_Qwen_GPU.Fattn_New
                          (Cap, Blk.Full.N_KV_Heads * Blk.Full.Head_Dim, Blk.Full.N_Q_Heads);
                     else
                        Hn := LLM_Qwen_GPU.Dnet_New
                          (Blk.DNet.N_V_Heads, Blk.DNet.Key_Head_Dim,
                           Blk.DNet.Value_Head_Dim, Blk.DNet.QKV_Out,
                           Shape (Blk.DNet.Conv_W) (2));
                     end if;
                     Handles (Bi * NL + (Li - 1)) := Interfaces.C.int (Hn);
                  end;
               end loop;
            end loop;
            --  warm
            LLM_Qwen_GPU.Chain_Forward_Batch
              (B, Rows'Address, Pos'Address, Handles'Address, Data_Address (Logits));
            TS := Ada.Real_Time.Clock;
            for Step in 1 .. NT loop
               LLM_Qwen_GPU.Chain_Forward_Batch
                 (B, Rows'Address, Pos'Address, Handles'Address, Data_Address (Logits));
               for Bi in 0 .. B - 1 loop Pos (Bi) := Interfaces.C.int (Step); end loop;
            end loop;
            declare
               Dt : constant Duration :=
                 Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - TS);
            begin
               Ada.Text_IO.Put_Line
                 (Ada.Text_IO.Standard_Error,
                  "[BATCH] B=" & Integer'Image (B)
                  & " steps=" & Integer'Image (NT)
                  & " wall=" & Float'Image (Float (Dt)) & "s"
                  & " AGGREGATE_tok_per_s="
                  & Float'Image (Float (B * NT) / Float (Dt))
                  & " per_lane=" & Float'Image (Float (NT) / Float (Dt)));
            end;

            --  Correctness: batched(B=1) must equal single-request on the same
            --  fresh state + input (the batched matvecs reduce to the single-row
            --  kernels at B=1). Compare the token-0 logits; max abs diff ~ 0.
            declare
               HS : array (0 .. NL - 1) of Interfaces.C.int := [others => 0];
               HB : array (0 .. NL - 1) of Interfaces.C.int := [others => 0];
               R1 : array (0 .. 0) of Interfaces.C.int := [0 => 0];
               P0 : array (0 .. 0) of Interfaces.C.int := [0 => 0];
               LS : constant Tensor := New_Tensor ([1, M.Vocab_Sz]);
               LB : constant Tensor := New_Tensor ([1, M.Vocab_Sz]);
               Max_Diff, Max_Mag : Float := 0.0;

               function Fresh (Li : Integer) return Integer is
                  Blk : LLM_Qwen_Blk.Qwen_Block renames M.Blocks (Li).all;
               begin
                  if Blk.Is_Full_Attn then
                     return LLM_Qwen_GPU.Fattn_New
                       (Cap, Blk.Full.N_KV_Heads * Blk.Full.Head_Dim, Blk.Full.N_Q_Heads);
                  else
                     return LLM_Qwen_GPU.Dnet_New
                       (Blk.DNet.N_V_Heads, Blk.DNet.Key_Head_Dim,
                        Blk.DNet.Value_Head_Dim, Blk.DNet.QKV_Out,
                        Shape (Blk.DNet.Conv_W) (2));
                  end if;
               end Fresh;
            begin
               for Li in 1 .. NL loop
                  HS (Li - 1) := Interfaces.C.int (Fresh (Li));
                  HB (Li - 1) := Interfaces.C.int (Fresh (Li));
               end loop;
               LLM_Qwen_GPU.Chain_Begin (HS'Address);
               LLM_Qwen_GPU.Chain_Forward (0, 0, HS'Address, Data_Address (LS));
               LLM_Qwen_GPU.Chain_End;
               LLM_Qwen_GPU.Chain_Forward_Batch
                 (1, R1'Address, P0'Address, HB'Address, Data_Address (LB));
               for K in 1 .. M.Vocab_Sz loop
                  Max_Diff := Float'Max (Max_Diff, abs (Get_Flat (LS, K) - Get_Flat (LB, K)));
                  Max_Mag  := Float'Max (Max_Mag, abs (Get_Flat (LS, K)));
               end loop;
               Ada.Text_IO.Put_Line
                 (Ada.Text_IO.Standard_Error,
                  "[BATCH-CHECK] max|logit_single - logit_batch(B=1)| ="
                  & Float'Image (Max_Diff) & "  (max|logit|=" & Float'Image (Max_Mag) & ")");
            end;
         end;
      end if;
   end Register_Chain;

   --  ---- Prefix KV-cache registry (phase 3) -------------------------------
   --  Maps a leading-system-prompt token prefix (64-bit FNV-1a hash + length)
   --  to the retained per-layer GPU snapshot slots captured after that prefix
   --  was prefilled once. A later request with the SAME system prompt restores
   --  those slots and prefills only the short varying suffix (user turn +
   --  assistant opener), turning a ~13.5s TTFT into ~sub-second. Gated by
   --  ASPIDA_PREFIX_CACHE and validated bit-exact (prefix-cache logits ==
   --  full-prefill logits) before it is trusted in serving.
   Prefix_Cache_On : constant Boolean :=
     Ada.Environment_Variables.Exists ("ASPIDA_PREFIX_CACHE");
   --  Per-request HIT/MISS logging — separate opt-in so the cache runs quietly
   --  in production (one journald line per LLM call is a lot under load).
   Prefix_Log_On : constant Boolean :=
     Ada.Environment_Variables.Exists ("ASPIDA_PREFIX_LOG");
   --  Multi-turn history cache: also snapshot at the end of each session turn
   --  (not just the system prompt) so a follow-up turn restores the whole
   --  conversation state and prefills only the new user message. DEFAULT (when
   --  the prefix cache is on); validated by the eval gate + multi-turn recall.
   --  ASPIDA_NO_PREFIX_HISTORY=1 falls back to system-prefix-only caching.
   Prefix_History_On : constant Boolean :=
     not Ada.Environment_Variables.Exists ("ASPIDA_NO_PREFIX_HISTORY");

   Max_Prefix_Layers  : constant := 128;   -- >= any model's block count
   --  Distinct system prompts cached. Each ~5.9k-token prefix snapshot costs
   --  ~290 MiB VRAM, so 8 caps the cache near ~2.3 GB — comfortable headroom
   --  under the resident 35B-Q8 model + per-lane K/V on a 48 GB card. Round-
   --  robin eviction recycles slots beyond this, so more agents still work
   --  (they just re-prefill on eviction) with no leak.
   Max_Prefix_Entries : constant := 8;

   type Slot_Storage is array (1 .. Max_Prefix_Layers) of Integer;
   Empty_Slots : constant Slot_Storage := [others => -1];

   type Key_Array  is array (1 .. Max_Prefix_Entries) of Interfaces.Unsigned_64;
   type Len_Array  is array (1 .. Max_Prefix_Entries) of Natural;
   type Bool_Array is array (1 .. Max_Prefix_Entries) of Boolean;
   type Snap_Array is array (1 .. Max_Prefix_Entries) of Slot_Storage;

   function Prefix_Hash (Ids : LLM_Tokenizer.Token_Array; N : Natural)
      return Interfaces.Unsigned_64
   is
      use type Interfaces.Unsigned_64;
      H : Interfaces.Unsigned_64 := 16#cbf29ce484222325#;  -- FNV-1a offset
   begin
      for I in Ids'First .. Ids'First + N - 1 loop
         H := (H xor Interfaces.Unsigned_64 (Ids (I))) * 16#100000001b3#;
      end loop;
      return H;
   end Prefix_Hash;

   use type Interfaces.Unsigned_64;

   --  Serialised registry. Reserve picks a target entry (recycling an evicted
   --  entry's CUDA slots so nothing leaks), the caller snapshots into those
   --  slots, then Commit stores the actual slot ids. Lookup returns a hit's
   --  slots for restore. Single-path prefill is already serialised by the
   --  server's Infer_Lock, but the protected object keeps it safe regardless.
   protected Prefix_Reg is
      procedure Lookup (Key : Interfaces.Unsigned_64; N : Natural;
                        Found : out Boolean; Idx : out Natural;
                        Slots : out Slot_Storage);
      procedure Release (Idx : Natural);
      procedure Reserve (Key : Interfaces.Unsigned_64; N : Natural;
                         Idx : out Natural; Target : out Slot_Storage);
      procedure Commit (Idx : Natural; Key : Interfaces.Unsigned_64;
                        N : Natural; Slots : Slot_Storage);
   private
      Keys   : Key_Array  := [others => 0];
      Lens   : Len_Array  := [others => 0];
      Valid  : Bool_Array := [others => False];
      Snaps  : Snap_Array := [others => Empty_Slots];
      Round  : Natural := 0;  -- round-robin eviction cursor
      Pins   : Len_Array := [others => 0];  -- in-flight restore/snapshot refs
   end Prefix_Reg;

   protected body Prefix_Reg is
      procedure Lookup (Key : Interfaces.Unsigned_64; N : Natural;
                        Found : out Boolean; Idx : out Natural;
                        Slots : out Slot_Storage) is
      begin
         Found := False; Idx := 0; Slots := Empty_Slots;
         for I in Keys'Range loop
            if Valid (I) and then Keys (I) = Key and then Lens (I) = N then
               Found := True; Idx := I; Slots := Snaps (I);
               Pins (I) := Pins (I) + 1;  -- pin until the caller Releases
               return;
            end if;
         end loop;
      end Lookup;

      procedure Release (Idx : Natural) is
      begin
         if Idx in Keys'Range and then Pins (Idx) > 0 then
            Pins (Idx) := Pins (Idx) - 1;
         end if;
      end Release;

      procedure Reserve (Key : Interfaces.Unsigned_64; N : Natural;
                         Idx : out Natural; Target : out Slot_Storage) is
      begin
         --  Reuse this exact key's entry if present AND not in flight.
         for I in Keys'Range loop
            if Keys (I) = Key and then Lens (I) = N and then Pins (I) = 0 then
               Idx := I; Target := Snaps (I); Valid (I) := False;
               Pins (I) := Pins (I) + 1;   -- pin until Commit/Release
               return;
            end if;
         end loop;
         --  Round-robin to an UNPINNED slot: never recycle CUDA slots a
         --  concurrent lane is mid-restore from (that was a use-after-free
         --  -> illegal memory access under the batcher).
         for K in 1 .. Max_Prefix_Entries loop
            Round := (Round mod Max_Prefix_Entries) + 1;
            if Pins (Round) = 0 then
               Idx := Round; Target := Snaps (Idx); Valid (Idx) := False;
               Pins (Idx) := Pins (Idx) + 1;
               return;
            end if;
         end loop;
         --  Every slot in flight (rare): skip snapshotting this turn.
         Idx := 0; Target := Empty_Slots;
      end Reserve;

      procedure Commit (Idx : Natural; Key : Interfaces.Unsigned_64;
                        N : Natural; Slots : Slot_Storage) is
      begin
         if Idx in Keys'Range then
            Keys (Idx) := Key; Lens (Idx) := N;
            Snaps (Idx) := Slots; Valid (Idx) := True;
         end if;
      end Commit;
   end Prefix_Reg;

   --  Ascending list of token offsets that are candidate prefix-cache
   --  boundaries: the leading system prompt AND (with ASPIDA_PREFIX_HISTORY)
   --  the end of every prior conversation turn. Decode_Tokens probes them
   --  longest-first to RESTORE the deepest cached state, and snapshots at the
   --  largest (the full history before the current user turn) so the next turn
   --  in the session chains onto it. Empty => caching disabled for the call.
   type Cache_Bounds_Array is array (Positive range <>) of Natural;

   function Decode_Tokens
     (M              : Qwen_Model;
      Prompt_Ids     : LLM_Tokenizer.Token_Array;
      Max_New_Tokens : Integer;
      Stop1, Stop2   : Integer;
      Sink           : access Token_Sink'Class;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access Gen_Stats := null;
      --  Length (in tokens, from Prompt_Ids'First) of the constant leading
      --  prefix (the agent's system prompt) that is eligible for the prefix
      --  KV-cache. 0 disables caching for this call. See Prefix_Reg.
      Prefix_Len     : Natural := 0;
      --  Multi-boundary cache offsets (ASPIDA_PREFIX_HISTORY). When non-empty,
      --  overrides Prefix_Len: probe these ascending offsets longest-first to
      --  restore the deepest cached turn, and snapshot at the largest. Empty =>
      --  fall back to the single Prefix_Len boundary (legacy behaviour).
      Cache_Bounds   : Cache_Bounds_Array := [1 .. 0 => 0];
      --  True when the prompt already opens a <think> block (forced thinking),
      --  so the reasoning-budget guard counts from token 0.
      Reason_Seeded  : Boolean := False) return String
   is
      Dim     : constant Integer := M.Model_Dim;
      Cap     : constant Integer :=
        Integer'Max (1, Prompt_Ids'Length + Max_New_Tokens);
      Out_Buf : Unbounded_String;
      Smp     : LLM_Sampler.Sampler := LLM_Sampler.Create (Params);
      Hist    : LLM_Sampler.History (1 .. Integer'Max (1, Max_New_Tokens)) :=
        [others => 0];
      N_Hist  : Natural := 0;
      Produced : Natural := 0;       -- generated tokens (completion_tokens)
      Hit_Stop : Boolean := False;   -- stopped on a stop token, not the cap

      --  Reasoning-budget guard. This fine-tune sometimes runs AWAY inside a
      --  <think> block — reasoning to the token cap and never emitting an
      --  answer (a 40 s+ empty reply; also floods the SSE buffer). Cap the
      --  chain-of-thought: once it has run this many tokens without closing,
      --  force-emit </think> so the model concludes and answers. 0 disables.
      Think_Open_Id  : constant Integer := LLM_Tokenizer.Token_To_Id (M.Tok, "<think>");
      Think_Close_Id : constant Integer := LLM_Tokenizer.Token_To_Id (M.Tok, "</think>");
      Think_Budget_Env : constant Natural :=
        (if Ada.Environment_Variables.Exists ("ASPIDA_THINK_BUDGET")
         then Natural'Value (Ada.Environment_Variables.Value ("ASPIDA_THINK_BUDGET"))
         --  2048, not 512: Ornith is an RL-trained coding reasoner whose hard
         --  answers routinely spend 1-3k tokens thinking (measured on the
         --  eval-hura hard set); 512 force-closed mid-thought and cost quality.
         --  Still a hard guard against run-away (0 disables entirely).
         else 2048);
      --  RESERVE THE ANSWER; do not ration it. The rule here used to hand
      --  reasoning 3/4 of Max_New_Tokens, which inverts the priority: a small
      --  request spends most of its budget thinking and has a quarter left to
      --  speak with. Measured on this model: max_tokens=128 gave reasoning 96
      --  tokens and the answer 32 -- exactly enough for "I'll write a detailed
      --  plan." and then finish_reason=length. EVERY budget behaved that way;
      --  the reply was always truncated, never finished. The same question with
      --  thinking off completes on its own in 561 tokens.
      --
      --  So: the answer is guaranteed Answer_Reserve tokens (or the entire
      --  budget when that is smaller), and reasoning gets only the surplus,
      --  still capped absolutely. At max_tokens >= 1024 this is identical to the
      --  old rule (512 thinking); it only stops the small-budget case from
      --  strangling itself.
      --
      --  llama.cpp keeps --reasoning-budget absolute and independent of
      --  n_predict: that avoids the strangling, but lets a small budget be spent
      --  entirely on thought and return an EMPTY answer. Reserving the answer
      --  avoids both failure modes.
      Answer_Reserve : constant Natural :=
        (if Ada.Environment_Variables.Exists ("ASPIDA_ANSWER_RESERVE")
         then Natural'Value (Ada.Environment_Variables.Value ("ASPIDA_ANSWER_RESERVE"))
         else 512);
      --  ASPIDA_THINK_BUDGET=0 disables the guard entirely (documented): the
      --  model then reasons until it closes the block itself.
      Think_Limit_On : constant Boolean := Think_Budget_Env > 0;
      --  0 here means "no room to think" -> close a stray <think> on its first
      --  token so the reply keeps the whole budget. It does NOT mean unlimited;
      --  Think_Limit_On carries that.
      Think_Budget   : constant Natural :=
        (if Max_New_Tokens > Answer_Reserve
         then Natural'Min (Think_Budget_Env, Max_New_Tokens - Answer_Reserve)
         else 0);
      In_Reason    : Boolean := Reason_Seeded;   -- prompt opened <think>?
      Reason_Start : Natural := 0;               -- Produced when reasoning began
      Think_Forced : Boolean := False;           -- already forced the close
      --  UTF-8 boundary tracking for the reasoning force-close. Forcing
      --  </think> in place of the sampled token would split a multi-byte
      --  character if the emitted bytes end mid-character (Ukrainian is 2
      --  bytes) — producing a broken char in reasoning_content. Pending is the
      --  continuation bytes still owed by the last lead byte (0 = on a
      --  boundary); the guard waits for a boundary before forcing, bounded by
      --  Force_Grace so a model that never completes a char still gets cut.
      Pending_UTF8 : Natural := 0;
      Force_Grace  : Natural := 0;

      --  Per-token host-tail profiler (env ASPIDA_TAIL_PROF): sampling vs stream.
      Tail_Prof : constant Boolean :=
        Ada.Environment_Variables.Exists ("ASPIDA_TAIL_PROF");
      Acc_Smp, Acc_Emt : Ada.Real_Time.Time_Span := Ada.Real_Time.Time_Span_Zero;
      Tail_N : Natural := 0;
      use type Ada.Real_Time.Time;
      use type Ada.Real_Time.Time_Span;

      --  Per-layer decode state (KV cache for full-attn, recurrent state +
      --  conv window for delta-net), threaded across tokens. One forward
      --  step costs O(1) matmuls instead of recomputing the sequence.
      Cache : array (1 .. M.N_Blocks) of LLM_Qwen_Blk.Block_State;

      --  Phase C chain: per-generation state handles + token position.
      Use_Chain : Boolean := False;
      Chain_Pos : Natural := 0;

      --  Continuous batching (env ASPIDA_BATCH_SERVE): when a lane is claimed,
      --  each token's forward is submitted to the shared batch Driver instead
      --  of running a serialised single-request forward.
      Batch_Mode : Boolean := False;
      My_Lane    : Integer := -1;
      Handles   : array (1 .. M.N_Blocks) of Interfaces.C.int :=
        [others => Interfaces.C.int (Integer'(-1))];

      procedure Free_States is
      begin
         if Batch_Mode then
            LLM_Batcher.End_Gen (My_Lane);
            My_Lane := -1; Batch_Mode := False;
         else
            LLM_Qwen_GPU.Chain_End;
         end if;
         --  Freeing device state mutates shared GPU vectors — serialise it in
         --  batch-serve mode (same reason as allocation).
         if LLM_Batcher.Enabled then LLM_Batcher.Alloc_Lock; end if;
         begin
            for I in Cache'Range loop
               if Cache (I).Is_Full then
                  LLM_Qwen_GPU.Fattn_Free (Cache (I).Full_St.GPU_Handle);
                  Cache (I).Full_St.GPU_Handle := -1;
               else
                  LLM_Qwen_GPU.Dnet_Free (Cache (I).DNet_St.GPU_Handle);
                  Cache (I).DNet_St.GPU_Handle := -1;
               end if;
            end loop;
         exception
            when others =>   -- A6: never leak Alloc_Lock on a teardown raise
               if LLM_Batcher.Enabled then LLM_Batcher.Alloc_Unlock; end if;
               raise;
         end;
         if LLM_Batcher.Enabled then LLM_Batcher.Alloc_Unlock; end if;
      end Free_States;

      --  One forward step under the shared step lock, released between steps
      --  (incl. on exception) so concurrent generations interleave per token.
      function Decode (Embed_Row : Integer) return Tensor is
         H  : Tensor := New_Tensor ([1, Dim]);
         TS : Ada.Real_Time.Time;
         Held_Step : Boolean := False;   -- A5: only Release the step lock we took
      begin
         --  Batched path: submit the step to the shared Driver (no step lock —
         --  the Driver is the sole GPU forward caller and serialises internally).
         if Batch_Mode then
            declare
               R : constant Tensor := New_Tensor ([1, M.Vocab_Sz]);
            begin
               if Chain_Pos >= Cap then
                  raise Constraint_Error with "chain KV overflow at"
                    & Integer'Image (Chain_Pos);
               end if;
               LLM_Batcher.Step
                 (My_Lane, Embed_Row - 1, Chain_Pos, M.N_Blocks,
                  Handles (1)'Address, Data_Address (R));
               Chain_Pos := Chain_Pos + 1;
               return R;
            end;
         end if;
         LLM_Step_Lock.Acquire;
         Held_Step := True;
         if Use_Chain then
            declare
               R : constant Tensor := New_Tensor ([1, M.Vocab_Sz]);
            begin
               if Chain_Pos >= Cap then
                  raise Constraint_Error with "chain KV overflow at"
                    & Integer'Image (Chain_Pos);
               end if;
               LLM_Qwen_GPU.Chain_Forward
                 (Embed_Row - 1, Chain_Pos, Handles (1)'Address,
                  Data_Address (R));
               Chain_Pos := Chain_Pos + 1;
               LLM_Step_Lock.Release; Held_Step := False;
               return R;
            end;
         end if;
         for D in 1 .. Dim loop
            Set_Flat (H, D, Get (M.Token_Emb, [Embed_Row, D]));
         end loop;
         TS := Ada.Real_Time.Clock;
         for I in 1 .. M.N_Blocks loop
            H := LLM_Qwen_Blk.Step (M.Blocks (I).all, Cache (I), H);
         end loop;
         if Prof2_On then
            Prof2_Loop := Prof2_Loop + Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - TS);
         end if;
         declare
            Normed : constant Tensor := LLM_RMSNorm.Forward (H, M.Final_Norm);
            R      : Tensor;
         begin
            TS := Ada.Real_Time.Clock;
            --  LM head is a dense F32 [vocab, dim] GEMV — offload it to the
            --  resident GPU dense matvec when the shim supports it, else the
            --  CPU pooled path. (The token embedding stays a cheap host gather.)
            if LLM_GPU.Has_Dense then
               R := New_Tensor ([1, Shape (M.LM_Head) (1)]);
               LLM_GPU.Dense_MatVec
                 (W_Addr  => Data_Address (M.LM_Head),
                  W_Bytes => Long_Long_Integer (Numel (M.LM_Head)) * 4,
                  In_Dim  => Shape (M.LM_Head) (2),
                  Out_Dim => Shape (M.LM_Head) (1),
                  X       => Data_Address (Normed),
                  Y       => Data_Address (R));
            else
               R := MatVec_Rows (M.LM_Head, Normed);
            end if;
            if Prof2_On then
               Prof2_Head := Prof2_Head + Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - TS);
               Prof2_Tick;
            end if;
            LLM_Step_Lock.Release; Held_Step := False;
            return R;
         end;
      exception
         when others =>
            if Held_Step then LLM_Step_Lock.Release; end if;
            raise;
      end Decode;

      Last_Logits : Tensor;
   begin
      --  Fail loud on an over-window prompt. Ctx_Len is the trained context;
      --  a prompt longer than it would drive RoPE past the range the model was
      --  trained on and return coherent-looking garbage with no error. Refuse
      --  instead: set Overflow (the server maps it to context_length_exceeded)
      --  and return nothing, before any forward runs.
      if Prompt_Ids'Length > M.Ctx_Len then
         if Stats /= null then
            Stats.all := (Prompt_Tokens     => Prompt_Ids'Length,
                          Completion_Tokens => 0,
                          Truncated         => False,
                          Overflow          => True);
         end if;
         return "";
      end if;

      --  Device-state allocation (Dnet_New/Fattn_New push to shared GPU vectors)
      --  is serialised in batch-serve mode, where several handler tasks set up
      --  generations concurrently.
      if LLM_Batcher.Enabled then LLM_Batcher.Alloc_Lock; end if;
      begin
         for I in 1 .. M.N_Blocks loop
            Cache (I) := LLM_Qwen_Blk.Init_State (M.Blocks (I).all, Cap);
         end loop;
         --  Phase C: use the resident chain when the model is registered and
         --  every layer's device state was allocated.
         Register_Chain (M);
         if LLM_Batcher.Enabled then LLM_Batcher.Alloc_Unlock; end if;
      exception
         when others =>
            if LLM_Batcher.Enabled then LLM_Batcher.Alloc_Unlock; end if;
            raise;
      end;
      Use_Chain := Chain_OK;
      for I in 1 .. M.N_Blocks loop
         declare
            Hn : constant Integer :=
              (if Cache (I).Is_Full then Cache (I).Full_St.GPU_Handle
               else Cache (I).DNet_St.GPU_Handle);
         begin
            if Hn < 0 then
               Use_Chain := False;
            end if;
            Handles (I) := Interfaces.C.int (Hn);
         end;
      end loop;
      --  CPU fallback path indexes the full host caches, which the fast
      --  Init_State stubbed out on layers that DID get a device handle —
      --  re-init those with Force_Host before any CPU Step runs.
      if not Use_Chain then
         for I in 1 .. M.N_Blocks loop
            Cache (I) := LLM_Qwen_Blk.Init_State
              (M.Blocks (I).all, Cap, Force_Host => True);
         end loop;
      end if;
      --  Claim a batch lane. Begin_Gen BLOCKS until one is free — falling back
      --  to the single path while the batcher is live would drive the shared
      --  resident chain state concurrently with the Driver (cross-generation
      --  KV corruption), so excess requests queue on the lane pool instead.
      if Use_Chain and then LLM_Batcher.Enabled then
         LLM_Batcher.Configure (M.N_Blocks, M.Vocab_Sz);
         LLM_Batcher.Begin_Gen (My_Lane);
         Batch_Mode := My_Lane >= 0;
      end if;
      if Use_Chain and then not Batch_Mode then
         LLM_Qwen_GPU.Chain_Begin (Handles (1)'Address);
      end if;

      --  Prefill (row = id + 1: ids are 0-based, embedding rows 1-based).
      --  Chunked GPU prefill when available: advance the resident state 32
      --  positions per call (matmuls batched over the chunk), instead of one
      --  token per forward — a 5-25k-token agent prompt went from 1-2 minutes
      --  (every request timed out) to seconds. Bit-identical to the per-token
      --  path. Falls back to per-token for the CPU path / empty prompt.
      if Prompt_Ids'Length = 0 then
         Last_Logits := Decode (1);
      elsif Use_Chain and then LLM_Qwen_GPU.Chain_Prefill_Available then
         declare
            --  Prefill chunk. The Q8 matmuls use the tensor-core kernel
            --  (weight-stationary), so a large chunk amortises the weight read
            --  ~4x better than 32. Must stay <= the CUDA-side PCH buffer cap.
            --  256 -> 512 -> 1024 (2026-07-15): larger chunks amortise the
            --  fixed per-chunk MoE expert-weight stream (1024 cuts MoE to
            --  57.7ms per 256-tok equivalent, was 155.8; measured ~10% faster
            --  prefill end-to-end). Runs with ASPIDA_PREFILL_SETS=1 (one scratch
            --  set) on the 46GB prod box, which co-hosts a 3.2GB voice_server.
            --  Measured prompt ceiling is >=20k tokens (no OOM), covering the
            --  platform's 25-100KB prompts, so a single 1024 scratch set
            --  (~440MB) is comfortably affordable. MUST match #define PCH in
            --  gpu/gpu_matvec.cu — see that comment for the full VRAM budget.
            PCHUNK : constant := 1024;
            Total  : constant Natural := Prompt_Ids'Length;
            Done   : Natural := 0;

            --  Prefix KV-cache (phase 3, single path only — the batcher path
            --  is phase 4). Active only under ASPIDA_PREFIX_CACHE, with a real
            --  system prefix that leaves a suffix to still prefill, when the
            --  shim exports the snapshot API. On a hit we restore the per-layer
            --  device state and jump Chain_Pos past the prefix; on a miss we
            --  prefill the prefix, snapshot every layer, then continue.
            --  Phase 4: also active under the batcher. Prefill (and thus the
            --  snapshot/restore) is per-lane via Chain_Prefill, which syncs the
            --  lane stream before returning, so the snapshot reads a complete
            --  state; the snapshot pools are mutex-guarded on the CUDA side and
            --  the registry is a protected object, so concurrent lanes are safe.
            --  Snapshot boundary: the full history before the current user
            --  turn (largest Cache_Bounds entry), else the single Prefix_Len.
            Snap_Len : constant Natural :=
              (if Cache_Bounds'Length > 0 then Cache_Bounds (Cache_Bounds'Last)
               else Prefix_Len);
            --  Don't spend a cache slot (there are only Max_Prefix_Entries) on a
            --  tiny prefix: an adaptive-effort classifier call carries only a
            --  ~100-400-token prompt that re-prefills in <0.5 s anyway, but if it
            --  snapshots it evicts the valuable ~6k agent-system-prompt snapshots
            --  in round-robin — so a session's greeting hits the cache for two
            --  turns then the classifier churns it out and turn 3+ cold-prefills
            --  the whole prompt again. Only cache prefixes worth the slot.
            Min_Snap : constant Natural := 1024;
            Cache_Active : constant Boolean :=
              Prefix_Cache_On and then Snap_Len >= Min_Snap
              and then Snap_Len < Total
              and then LLM_Qwen_GPU.Prefix_Cache_Available;
            Hit         : Boolean := False;
            Restore_Len : Natural := 0;
            RSlots      : Slot_Storage := Empty_Slots;   -- restore FROM
            SSlots      : Slot_Storage := Empty_Slots;   -- snapshot INTO
            Res_Idx     : Natural := 0;
            Hit_Idx     : Natural := 0;
            Snap_Key    : Interfaces.Unsigned_64 := 0;
            Need_Snap   : Boolean := False;
            --  A2 (cert): a client stream abort raises Socket_Error from
            --  Sink.Tick INSIDE the prefill loop, between a pin (Lookup hit /
            --  Reserve) and its Release. Without exception-safe release the
            --  slot stays Pins>0/Valid=false forever -> prefix cache dies +
            --  ~290MB VRAM stranded per abort. Track pins, release on unwind.
            Hit_Pinned  : Boolean := False;
            Res_Pinned  : Boolean := False;
         begin
            Last_Logits := New_Tensor ([1, M.Vocab_Sz]);

            if Cache_Active then
               --  Probe restore boundaries longest-first: restore the deepest
               --  already-cached turn. History mode probes every turn boundary;
               --  legacy mode the single system prefix.
               if Cache_Bounds'Length > 0 then
                  for I in reverse Cache_Bounds'Range loop
                     declare
                        B : constant Natural := Cache_Bounds (I);
                     begin
                        if B > 0 and then B < Total then
                           Prefix_Reg.Lookup
                             (Prefix_Hash (Prompt_Ids, B), B, Hit, Hit_Idx, RSlots);
                           if Hit then Restore_Len := B; exit; end if;
                        end if;
                     end;
                  end loop;
               else
                  Prefix_Reg.Lookup
                    (Prefix_Hash (Prompt_Ids, Snap_Len), Snap_Len, Hit, Hit_Idx, RSlots);
                  if Hit then Restore_Len := Snap_Len; end if;
               end if;
               Hit_Pinned := Hit;   -- Lookup pins only on hit

               --  Snapshot the full-history boundary unless it is already cached
               --  (a full hit at Snap_Len). Reserve a distinct slot set to write.
               Need_Snap := Restore_Len < Snap_Len;
               if Need_Snap then
                  Snap_Key := Prefix_Hash (Prompt_Ids, Snap_Len);
                  Prefix_Reg.Reserve (Snap_Key, Snap_Len, Res_Idx, SSlots);
                  if Res_Idx = 0 then Need_Snap := False; else Res_Pinned := True; end if;
               end if;
            end if;

            if Hit then
               --  Restore each layer's snapshotted state, then skip to Restore_Len.
               for I in 1 .. M.N_Blocks loop
                  declare
                     H : constant Integer := Integer (Handles (I));
                     R : constant Integer :=
                       (if Cache (I).Is_Full
                        then LLM_Qwen_GPU.Fattn_Restore (H, RSlots (I))
                        else LLM_Qwen_GPU.Dnet_Restore (H, RSlots (I)));
                  begin
                     if R < 0 then
                        raise Constraint_Error
                          with "prefix restore failed at layer"
                               & Integer'Image (I);
                     end if;
                  end;
               end loop;
               Chain_Pos := Restore_Len;
               Done := Restore_Len;
               Prefix_Reg.Release (Hit_Idx);  -- restores done; unpin source
               Hit_Pinned := False;
               if Prefix_Log_On then
                  Ada.Text_IO.Put_Line
                    (Ada.Text_IO.Standard_Error,
                     "[PREFIXCACHE] HIT restore=" & Restore_Len'Image
                     & " snap=" & Snap_Len'Image
                     & " suffix=" & Natural'Image (Total - Restore_Len));
               end if;
            end if;

            while Done < Total loop
               declare
                  --  Stop the chunk exactly at Snap_Len while a snapshot is still
                  --  pending, so we can capture the state at that boundary.
                  Limit : constant Natural :=
                    (if Cache_Active and then Need_Snap and then Done < Snap_Len
                     then Snap_Len - Done else Total - Done);
                  P    : constant Natural := Integer'Min (PCHUNK, Limit);
                  Rows : array (1 .. P) of Interfaces.C.int;
               begin
                  if Chain_Pos + P > Cap then
                     raise Constraint_Error with "chain KV overflow at"
                       & Integer'Image (Chain_Pos);
                  end if;
                  for J in 1 .. P loop
                     Rows (J) := Interfaces.C.int
                       (Prompt_Ids (Prompt_Ids'First + Done + J - 1));
                  end loop;
                  LLM_Qwen_GPU.Chain_Prefill
                    ((if Batch_Mode then My_Lane else 0), P,
                     Rows (1)'Address, Chain_Pos,
                     Handles (1)'Address, Data_Address (Last_Logits));
                  Chain_Pos := Chain_Pos + P;
                  Done := Done + P;
                  if Sink /= null then
                     Sink.Tick;
                  end if;
               end;

               --  Reached the snapshot boundary: snapshot every layer into the
               --  reserved slots and commit, so the NEXT turn in this session
               --  restores it and prefills only the new user message.
               if Cache_Active and then Need_Snap and then Done = Snap_Len then
                  declare
                     Actual : Slot_Storage := Empty_Slots;
                     All_Ok : Boolean := True;
                  begin
                     for I in 1 .. M.N_Blocks loop
                        declare
                           H : constant Integer := Integer (Handles (I));
                        begin
                           Actual (I) :=
                             (if Cache (I).Is_Full
                              then LLM_Qwen_GPU.Fattn_Snapshot
                                     (H, Snap_Len, SSlots (I))
                              else LLM_Qwen_GPU.Dnet_Snapshot (H, SSlots (I)));
                           if Actual (I) < 0 then All_Ok := False; end if;
                        end;
                     end loop;
                     if All_Ok then
                        Prefix_Reg.Commit (Res_Idx, Snap_Key, Snap_Len, Actual);
                     end if;
                     Prefix_Reg.Release (Res_Idx);  -- snapshot done; unpin
                     Res_Pinned := False;
                     Need_Snap := False;
                     if Prefix_Log_On then
                        Ada.Text_IO.Put_Line
                          (Ada.Text_IO.Standard_Error,
                           "[PREFIXCACHE] SNAP n=" & Snap_Len'Image
                           & " restored=" & Restore_Len'Image
                           & " ok=" & All_Ok'Image);
                     end if;
                  end;
               end if;
            end loop;
         exception
            when others =>
               --  Exception-safe pin release (A2): never leak a prefix pin.
               if Res_Pinned then Prefix_Reg.Release (Res_Idx); Res_Pinned := False; end if;
               if Hit_Pinned then Prefix_Reg.Release (Hit_Idx); Hit_Pinned := False; end if;
               raise;
         end;
      else
         for I in Prompt_Ids'Range loop
            Last_Logits := Decode (Prompt_Ids (I) + 1);
            if Sink /= null then
               Sink.Tick;
            end if;
         end loop;
      end if;

      for Step in 1 .. Max_New_Tokens loop
         --  Min-token floor: while fewer than Min_Tokens have been produced,
         --  drive the stop-token logits to -inf so the sampler cannot draw
         --  them. This defeats the "im_end on the very first token" pathology
         --  (some Qwen3 reasoning fine-tunes emit 0-token answers on certain
         --  prompts). Last_Logits is a fresh tensor each step (from Decode),
         --  so mutating it here corrupts no cached state.
         if Params.Min_Tokens > 0 and then Produced < Params.Min_Tokens then
            if Stop1 >= 0 then
               Set_Flat (Last_Logits, Stop1 + 1, -1.0e30);
            end if;
            if Stop2 >= 0 and then Stop2 /= Stop1 then
               Set_Flat (Last_Logits, Stop2 + 1, -1.0e30);
            end if;
         end if;
         declare
            Win : constant Natural :=
              Integer'Min (N_Hist, Integer'Max (0, Params.Repeat_Last_N));
            T_Smp : constant Ada.Real_Time.Time :=
              (if Tail_Prof then Ada.Real_Time.Clock else Ada.Real_Time.Time_First);
            Tid : Integer := LLM_Sampler.Next
              (Smp, Last_Logits, Hist (N_Hist - Win + 1 .. N_Hist));
            Best_Row : Integer;
         begin
            --  Reasoning-budget guard: if a <think> block has run past the
            --  budget without closing, force this token to </think> so the
            --  model concludes and produces the answer (prevents the runaway
            --  empty reply). Then track reasoning state from the token in play.
            if Think_Limit_On and then Think_Close_Id >= 0 then
               if In_Reason and then not Think_Forced
                 and then Produced - Reason_Start >= Think_Budget
                 and then Tid /= Think_Close_Id
               then
                  if Pending_UTF8 = 0 or else Force_Grace >= 4 then
                     Tid := Think_Close_Id;
                     Think_Forced := True;
                  else
                     --  Mid-multi-byte-character: let this token through to
                     --  finish the char, then force on a later step. Bounded by
                     --  Force_Grace (a char is <= 4 bytes) so this never hangs.
                     Force_Grace := Force_Grace + 1;
                  end if;
               end if;
               if Tid = Think_Open_Id then
                  In_Reason := True; Reason_Start := Produced; Think_Forced := False;
               elsif Tid = Think_Close_Id then
                  In_Reason := False;
               end if;
            end if;
            Best_Row := Tid + 1;   -- 1-based embedding row
            if Tail_Prof then
               Acc_Smp := Acc_Smp + (Ada.Real_Time.Clock - T_Smp);
               Tail_N := Tail_N + 1;
               if Tail_N mod 50 = 0 then
                  Ada.Text_IO.Put_Line
                    (Ada.Text_IO.Standard_Error,
                     "[TAILPROF] sample avg" & Float'Image
                       (Float (Ada.Real_Time.To_Duration (Acc_Smp))
                        / Float (Tail_N) * 1000.0)
                     & " ms | stream avg" & Float'Image
                       (Float (Ada.Real_Time.To_Duration (Acc_Emt))
                        / Float (Tail_N) * 1000.0)
                     & " ms (n=" & Natural'Image (Tail_N) & ")");
               end if;
            end if;
            exit when Best_Row < 1 or else Best_Row > M.Vocab_Sz;
            if Tid = Stop1 or else Tid = Stop2 then         -- natural stop
               Hit_Stop := True;
               exit;
            end if;
            declare
               Piece : constant String := LLM_Tokenizer.Decode_One (M.Tok, Tid);
            begin
               Append (Out_Buf, Piece);
               --  Advance the UTF-8 boundary state over this token's bytes so
               --  the reasoning force-close above never lands mid-character.
               for I in Piece'Range loop
                  declare
                     B : constant Natural := Character'Pos (Piece (I));
                  begin
                     if B >= 16#80# and then B < 16#C0# then   -- continuation
                        if Pending_UTF8 > 0 then
                           Pending_UTF8 := Pending_UTF8 - 1;
                        end if;
                     elsif B < 16#80# then Pending_UTF8 := 0;   -- ASCII
                     elsif B < 16#E0# then Pending_UTF8 := 1;   -- 2-byte lead
                     elsif B < 16#F0# then Pending_UTF8 := 2;   -- 3-byte lead
                     else                  Pending_UTF8 := 3;   -- 4-byte lead
                     end if;
                  end;
               end loop;
               if Sink /= null then            -- stream this token now
                  declare
                     T_Emt : constant Ada.Real_Time.Time :=
                       (if Tail_Prof then Ada.Real_Time.Clock
                        else Ada.Real_Time.Time_First);
                  begin
                     Sink.Emit (Piece);
                     if Tail_Prof then
                        Acc_Emt := Acc_Emt + (Ada.Real_Time.Clock - T_Emt);
                     end if;
                  end;
               end if;
            end;
            Produced := Produced + 1;
            N_Hist := N_Hist + 1; Hist (N_Hist) := Tid;
            exit when Step = Max_New_Tokens;
            --  Client disconnected mid-stream: stop HERE via a clean loop exit
            --  (Free_States runs on the normal path) rather than letting the
            --  sink raise and tear the batch lane down mid-decode, which
            --  corrupted shared GPU state and crashed the next generation.
            exit when Sink /= null and then Sink.Stop_Requested;
            Last_Logits := Decode (Best_Row);
         end;
      end loop;

      if Stats /= null then
         Stats.all := (Prompt_Tokens     => Prompt_Ids'Length,
                       Completion_Tokens => Produced,
                       Truncated         => not Hit_Stop,
                       Overflow          => False);
      end if;
      Free_States;
      return To_String (Out_Buf);
   exception
      when others =>
         Free_States;
         raise;
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

   function One (Id : Integer) return LLM_Tokenizer.Token_Array is
     [1 => Id];

   --  Special-token-aware tokenisation for chat message bodies. The Hura /
   --  Qwen3.5 GGUF carries <think>, </think>, <tool_call>, </tool_call> as
   --  single "added" vocab ids (e.g. <think>=248068), but byte-level Encode
   --  splits each into 3-4 ordinary BPE tokens. Feeding those splits, the model
   --  never saw the real control tokens: it ignored the platform's empty
   --  <think></think> "thinking-done" prefill, reasoned anyway, and — with no
   --  in-window repeat guard on block-level text — looped in reasoning until the
   --  token cap, emitting ZERO answer tokens ("empty response"). We resolve each
   --  known special to its id and byte-encode only the gaps, matching
   --  llama.cpp / Ollama's parse_special path. <|im_start|>/<|im_end|> are
   --  intentionally NOT in the set: turn frames are added structurally via
   --  One (Im_Start_Id), so leaving them literal in a body blocks turn-boundary
   --  injection from message text.
   function Encode_Chat (M : Qwen_Model; S : String)
      return LLM_Tokenizer.Token_Array
   is
      use type LLM_Tokenizer.Token_Array;

      --  Longest special starting exactly at S (P); 0 if none, with its id.
      function Special_At (P : Positive; Id : out Integer) return Natural is
         function Try (Lit : String) return Natural is
         begin
            if P + Lit'Length - 1 <= S'Last
              and then S (P .. P + Lit'Length - 1) = Lit
            then
               declare
                  T : constant Integer := LLM_Tokenizer.Token_To_Id (M.Tok, Lit);
               begin
                  if T >= 0 then
                     Id := T;
                     return Lit'Length;
                  end if;
               end;
            end if;
            return 0;
         end Try;
         L : Natural;
      begin
         Id := -1;
         --  longest first so "</tool_call>" beats "<tool_call>" etc.
         L := Try ("</tool_call>"); if L > 0 then return L; end if;
         L := Try ("<tool_call>");  if L > 0 then return L; end if;
         L := Try ("</think>");     if L > 0 then return L; end if;
         L := Try ("<think>");      if L > 0 then return L; end if;
         return 0;
      end Special_At;
   begin
      for P in S'Range loop
         declare
            Id : Integer;
            L  : constant Natural := Special_At (P, Id);
         begin
            if L > 0 then
               return LLM_Tokenizer.Encode (M.Tok, S (S'First .. P - 1))
                      & One (Id)
                      & Encode_Chat (M, S (P + L .. S'Last));
            end if;
         end;
      end loop;
      return LLM_Tokenizer.Encode (M.Tok, S);
   end Encode_Chat;

   --  True when S is an EMPTY reasoning prefill: "<think></think>" with only
   --  whitespace around/inside it. The platform api appends exactly this as a
   --  trailing assistant message (THINKING_PREFILL_CONTENT = "<think></think>
   --  \n\n"). On Ollama that prefill is a NO-OP — the hura Modelfile template
   --  re-opens thinking on every turn — so the model always shows its
   --  reasoning. We match that: an empty-think prefill is rendered as an OPEN
   --  <think> at the assistant turn (see Chat_Raw) so the model reasons, and
   --  the parser is seeded in reasoning to stream those thoughts.
   function Is_Empty_Think (S : String) return Boolean is
      Compact : String (1 .. S'Length);
      N       : Natural := 0;
   begin
      for I in S'Range loop
         if S (I) not in ' ' | ASCII.LF | ASCII.CR | ASCII.HT then
            N := N + 1;
            Compact (N) := S (I);
         end if;
      end loop;
      return Compact (1 .. N) = "<think></think>";
   end Is_Empty_Think;

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
      if Msg.Img_Ntok > 0 then
         return One (M.Im_Start_Id)
           & LLM_Tokenizer.Encode (M.Tok, Role_Str (Msg.Role) & LF)
           & One (248053)
           & LLM_Tokenizer.Token_Array'(1 .. Msg.Img_Ntok => 248056)
           & One (248054)
           & Encode_Chat (M, To_String (Msg.Text))
           & One (M.Im_End_Id)
           & LLM_Tokenizer.Encode (M.Tok, LF);
      end if;
      return One (M.Im_Start_Id)
        & LLM_Tokenizer.Encode (M.Tok, Role_Str (Msg.Role) & LF)
        & Encode_Chat (M, To_String (Msg.Text))
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

   --  Structured chat internal: build the prompt, generate raw pieces,
   --  drive the FSM parser, and return the assembled Chat_Result. Used by
   --  both the streaming (with Chat_Sink) and non-streaming variants.
   --  A4 (cert): vision state (Vis_Vtok/Ntok/Active + the single C-side
   --  Set_Vision slot) is package-global, shared across handler tasks. Two
   --  concurrent image requests would race: task B's Prep_Vision (Clear_Vision
   --  + overwrite the globals) during task A's Setup_Vision->prefill window ->
   --  A prefills with B's visual tokens. Serialize the vision span per request
   --  with a discriminated RAII holder: text requests pass Active => False and
   --  never touch the gate, so the hot path is unaffected; concurrent IMAGE
   --  requests serialize (Finalize releases on every exit, incl. exception).
   protected Vision_Serial is
      entry Acquire;
      procedure Release;
   private
      Busy : Boolean := False;
   end Vision_Serial;

   protected body Vision_Serial is
      entry Acquire when not Busy is begin Busy := True; end Acquire;
      procedure Release is begin Busy := False; end Release;
   end Vision_Serial;

   type Vision_Hold (Active : Boolean) is
     new Ada.Finalization.Limited_Controlled with null record;
   overriding procedure Initialize (H : in out Vision_Hold);
   overriding procedure Finalize   (H : in out Vision_Hold);
   overriding procedure Initialize (H : in out Vision_Hold) is
   begin
      if H.Active then Vision_Serial.Acquire; end if;
   end Initialize;
   overriding procedure Finalize (H : in out Vision_Hold) is
   begin
      if H.Active then Vision_Serial.Release; end if;
   end Finalize;

   --  Native Ornith vision (one image/request). Prep_Vision runs the ViT and
   --  marks the image message with its visual-token count; Setup_Vision finds
   --  the <|image_pad|> rows in the built prompt and hands the visual tokens to
   --  the GPU for prefill injection. Guarded: no image => no-op, chat bit-exact.
   Vis_Max_Tok : constant := 8192;
   Vis_Vtok    : array (0 .. Vis_Max_Tok * 2048 - 1) of Interfaces.C.C_float;
   Vis_Ntok    : Natural := 0;
   Vis_Active  : Boolean := False;

   function Prep_Vision (Conv_In : Message_Array) return Message_Array is
      Result : Message_Array := Conv_In;
      GH, GW : aliased Integer := 0;
   begin
      Vis_Ntok := 0; Vis_Active := False;
      --  Clear any stale C-side vision state so it can never leak into this
      --  request's prefill (prior request's tokens, an aborted decode, or the
      --  vision warm-up). Image requests re-arm it via Setup_Vision below.
      LLM_Qwen_GPU.Clear_Vision;
      if not LLM_Qwen_GPU.Vision_Available then return Result; end if;
      for I in Result'Range loop
         if Result (I).Image /= Ada.Strings.Unbounded.Null_Unbounded_String then
            declare
               N : constant Integer := LLM_Qwen_GPU.Vit_From_B64
                 (To_String (Result (I).Image), Vis_Vtok'Address, GH'Access, GW'Access);
            begin
               if N > 0 and then N <= Vis_Max_Tok then
                  Result (I).Img_Ntok := N; Vis_Ntok := N; Vis_Active := True;
               end if;
            end;
            exit;
         end if;
      end loop;
      return Result;
   end Prep_Vision;

   function Setup_Vision (Ids : LLM_Tokenizer.Token_Array) return Boolean is
      Positions : array (1 .. Natural'Max (Vis_Ntok, 1)) of Interfaces.C.int;
      K : Natural := 0;
   begin
      if not Vis_Active or else Vis_Ntok = 0 then return False; end if;
      for I in Ids'Range loop
         if Ids (I) = 248056 then
            K := K + 1;
            if K <= Vis_Ntok then Positions (K) := Interfaces.C.int (I - Ids'First); end if;
         end if;
      end loop;
      if K = Vis_Ntok then
         LLM_Qwen_GPU.Set_Vision (Vis_Ntok, Positions'Address, Vis_Vtok'Address);
      end if;
      return True;
   end Setup_Vision;

   function Chat_Raw
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer;
      Sink : access Chat_Sink'Class;
      Params : LLM_Sampler.Params;
      Stats : access Gen_Stats) return Chat_Result
   is
      --  Don't start a thought there is no room to finish. Decode_Tokens
      --  reserves ASPIDA_ANSWER_RESERVE tokens for the reply and gives
      --  reasoning only the surplus; when the surplus is zero, letting the
      --  model open <think> at all just burns tokens that get force-closed a
      --  moment later and reported as reasoning_content the caller did not ask
      --  for. Prefill the canonical closed block instead (the same thing
      --  Ollama's `think:false` does) so the whole budget is the answer's.
      --
      --  Measured: this question with max_tokens=128 produced 96 tokens of
      --  discarded reasoning and a 32-token stub; with thinking off the same
      --  128 tokens are all answer.
      Answer_Reserve : constant Natural :=
        (if Ada.Environment_Variables.Exists ("ASPIDA_ANSWER_RESERVE")
         then Natural'Value (Ada.Environment_Variables.Value ("ASPIDA_ANSWER_RESERVE"))
         else 512);
      No_Room_To_Think : constant Boolean :=
        Params.Enable_Thinking and then Max_New_Tokens <= Answer_Reserve;
      Params_Eff : constant LLM_Sampler.Params :=
        (if No_Room_To_Think
         then (Params with delta Enable_Thinking => False)
         else Params);

      LF     : constant String := [1 => ASCII.LF];
      --  The platform api ends the conversation with an assistant `<think></think>`
      --  prefill. On Ollama that is a no-op and the hura template re-opens
      --  thinking every turn, so the model always reasons visibly. We match it:
      --  render that prefill as an OPEN `<think>` (see the Ids build) and seed
      --  the parser in reasoning so the chain-of-thought streams as
      --  reasoning_content instead of being mis-read as the answer.
      Last_Is_Asst : constant Boolean :=
        Conversation'Length > 1
        and then Conversation (Conversation'Last).Role = Role_Assistant;
      Empty_Think  : constant Boolean :=
        Last_Is_Asst
        and then Is_Empty_Think (To_String (Conversation (Conversation'Last).Text));
      --  Ornith-1.0 is an RL-trained REASONING model: its benchmark quality
      --  (SWE-Bench/Terminal-Bench) is achieved WITH thinking, and its official
      --  chat template ALWAYS ends the generation prompt with an OPEN
      --  "<think>\n" (enable_thinking=false is the exception, not the rule).
      --  The old default suppressed thinking because reasoning could run away
      --  to the token cap — that blocker is gone: Decode_Tokens carries a
      --  think-budget guard (ASPIDA_THINK_BUDGET force-close + answer reserve).
      --  So thinking is now ON by default for BOTH the platform's empty
      --  <think></think> prefill and fresh turns, matching the official
      --  template exactly. Opt out with ASPIDA_NO_FORCE_THINK (or per-request
      --  enable_thinking=false, which prefills the canonical closed block).
      Think_Open   : constant Boolean :=
        Params_Eff.Enable_Thinking
        and then (Empty_Think or else not Last_Is_Asst)
        and then not Ada.Environment_Variables.Exists ("ASPIDA_NO_FORCE_THINK");
      P      : aliased LLM_Chat_Parser.Parser :=
        LLM_Chat_Parser.New_Parser (Start_In_Reasoning => Think_Open);
      Nullish : constant Null_Sink_Access := New_Null_Sink;
      function Resolve_Sink return access Chat_Sink'Class is
      begin
         if Sink /= null then
            return Sink;
         else
            return Nullish.all'Access;
         end if;
      end Resolve_Sink;
      SinkRef : constant access Chat_Sink'Class := Resolve_Sink;

      --  Live-streaming bridge: as Decode_Tokens produces each token it calls
      --  Emit here, which feeds that token straight into the chat FSM parser,
      --  so On_Reasoning / On_Text / On_Tool_Call fire on SinkRef AS tokens
      --  arrive — true token-by-token streaming.
      --
      --  Replaces a generate-everything-then-parse design (Decode_Tokens with a
      --  null sink, then one Feed over the whole Raw string): time-to-first-token
      --  was the full generation time, a disconnected client was not noticed
      --  until generation finished, and the handler kept a GPU busy producing
      --  tokens nobody would read — the shape behind the 2026-07-13
      --  disconnect-storm hang. Streaming live also gives cancellation for free:
      --  when SinkRef writes to a dead client it raises, Decode_Tokens frees its
      --  GPU state and re-raises, and generation stops within one token.
      --
      --  Declared local to Chat_Raw so its anonymous access components may point
      --  at the local parser P.
      type Parser_Bridge is new Token_Sink with record
         Psr : access LLM_Chat_Parser.Parser;
         Tgt : access Chat_Sink'Class;
      end record;
      overriding procedure Emit (S : in out Parser_Bridge; Piece : String);
      overriding procedure Tick (S : in out Parser_Bridge);

      overriding procedure Emit (S : in out Parser_Bridge; Piece : String) is
      begin
         LLM_Chat_Parser.Feed (S.Psr.all, Piece, S.Tgt);
      end Emit;

      overriding procedure Tick (S : in out Parser_Bridge) is
      begin
         --  Forward prefill heartbeats so the downstream sink keeps the channel
         --  warm during prompt-eval (before the first token).
         if S.Tgt /= null then
            S.Tgt.Tick;
         end if;
      end Tick;
   begin
      --  Fall back to raw generation when the model has no ChatML tokens.
      if M.Im_Start_Id < 0 or else M.Im_End_Id < 0 then
         declare
            Last : constant Unbounded_String :=
              Conversation (Conversation'Last).Text;
            Raw  : constant String :=
              Generate (M, To_String (Last), Max_New_Tokens, null);
         begin
            LLM_Chat_Parser.Feed (P, Raw, SinkRef);
            LLM_Chat_Parser.Finalize (P, SinkRef);
            declare
               NC : constant Natural := LLM_Chat_Parser.N_Tool_Calls (P);
            begin
               return R : Chat_Result (NC) do
                  R.Reasoning := To_Unbounded_String
                    (LLM_Chat_Parser.Reasoning_Of (P));
                  R.Answer    := To_Unbounded_String
                    (LLM_Chat_Parser.Answer_Of (P));
                  R.Finish    := To_Unbounded_String
                    (LLM_Chat_Parser.Finish_Of (P));
                  declare
                     Parsed : constant LLM_Chat_Parser.Tool_Call_Array :=
                       LLM_Chat_Parser.Tool_Calls_Of (P);
                  begin
                     for I in 1 .. NC loop
                        R.Tool_Calls (I) :=
                          (Id           => Parsed (I).Id,
                           Name         => Parsed (I).Name,
                           Arguments_JS => Parsed (I).Arguments_JS);
                     end loop;
                  end;
               end return;
            end;
         end;
      end if;

      declare
         use type LLM_Tokenizer.Token_Array;
         --  Prefill only the assistant turn opener. Do NOT prefill any
         --  reasoning marker: the chat parser starts in S_Idle and detects
         --  the reasoning opener (canonical ąd OR Hura's bare "think") in
         --  the GENERATED stream, using a balance counter where the first
         --  occurrence opens and the second closes a region. Prefilling an
         --  opener puts it in the prompt (not the generated stream), so the
         --  parser never sees it and the balance is thrown off. The earlier
         --  code prefilled "think\n\nthink\n\n" (open+close of an EMPTY
         --  block): that suppressed reasoning and primed the model with the
         --  literal word "think" repeated, which made Hura loop on
         --  "think\n…" or emit EOS immediately on prompts such as the UA
         --  recursion question (greedy, with a system prompt -> 0 tokens).
         --  Letting the model emit its own tags restores reasoning and fixes
         --  the degenerate stops.
         --  Enable_Thinking=False (Ollama `think:false`): prefill the CANONICAL
         --  closed empty think block, exactly as Qwen3.6's own chat template
         --  does for no-think — the model then answers directly. (This is the
         --  angle-bracketed form; the earlier degenerate loop came from bare
         --  "think\n\nthink\n\n" without brackets, which primed the literal word.)
         --  If the caller ends the conversation with an ASSISTANT message it is
         --  a PREFILL (the platform api sends a leading <think></think> assistant
         --  turn to steer reasoning): open that assistant turn and let the model
         --  continue it — do NOT close it with <|im_end|> and then open a SECOND
         --  assistant turn. The old code rendered every message closed and always
         --  appended a fresh opener, so a trailing assistant produced a DOUBLE
         --  assistant turn (…assistant\n<think></think><|im_end|>\n<|im_start|>
         --  assistant\n) — a malformed prompt that made the model hallucinate and
         --  loop. Ollama treats a trailing assistant message as a prefill; match it.
         --  Prefix over messages before any trailing assistant prefill.
         Has_Image : constant Boolean :=
           (for some Msg of Conversation =>
              Msg.Image /= Ada.Strings.Unbounded.Null_Unbounded_String);
         Vis_Guard : Vision_Hold (Active => Has_Image);   -- A4: serialize image reqs
         pragma Unreferenced (Vis_Guard);
         Conv : constant Message_Array := Prep_Vision (Conversation);
         Prefix : constant LLM_Tokenizer.Token_Array :=
           (if Last_Is_Asst then
              Conv_Ids (M, Conv (Conv'First .. Conv'Last - 1), Conv'First)
            else
              Conv_Ids (M, Conv, Conv'First));

         --  Prefix KV-cache boundary: the leading run of Role_System messages
         --  (the agent's constant system prompt + synthesized tools block).
         --  Its token count is the first N tokens of Ids (Conv_Ids concatenates
         --  per message, so Conv_Ids over the system run is a true prefix of the
         --  full Ids). Only computed under ASPIDA_PREFIX_CACHE — the extra
         --  tokenisation is paid only when the cache is enabled.
         function Sys_End return Natural is
            E : Natural := Conversation'First - 1;
         begin
            for I in Conversation'Range loop
               exit when Conversation (I).Role /= Role_System;
               E := I;
            end loop;
            return E;
         end Sys_End;
         Sys_Prefix_Len : constant Natural :=
           (if Prefix_Cache_On and then Sys_End >= Conversation'First
            then Conv_Ids
                   (M, Conversation (Conversation'First .. Sys_End),
                    Conversation'First)'Length
            else 0);

         --  History-cache boundaries: the cumulative token offset at the end of
         --  each message up to (but excluding) any trailing assistant prefill —
         --  i.e. every session-turn boundary. Decode_Tokens restores the deepest
         --  cached one and snapshots the largest (the whole conversation before
         --  the current user turn). Only built when ASPIDA_PREFIX_HISTORY is on.
         function Hist_Bounds return Cache_Bounds_Array is
            Last_Msg : constant Integer :=
              Conversation'Last - (if Last_Is_Asst then 1 else 0);
            --  Snapshot only up to the last ASSISTANT message (end of the
            --  STABLE persisted history), never the current turn. The platform
            --  now emits per-turn dynamic context (retrieved memory, tenant
            --  status) as a system message immediately BEFORE the user turn, so
            --  the tail [dynamic-system, current-user] is ephemeral — it is not
            --  in next turn's history. Snapshotting the full prefix would bake
            --  that ephemeral tail into the key and break the turn-to-turn
            --  chain (next turn's history has [user, assistant] where the
            --  snapshot had [dynamic, user]). Stopping at the last assistant
            --  keeps the snapshot byte-stable; the current turn re-prefills as
            --  the suffix. No assistant yet (turn 1) => snapshot the leading
            --  message (the stable system prompt).
            Last_Stable : Integer := Conversation'First;
            Tmp : Cache_Bounds_Array (1 .. 64) := [others => 0];
            NB  : Natural := 0;
         begin
            if not (Prefix_Cache_On and then Prefix_History_On)
              or else Last_Msg < Conversation'First
            then
               return [1 .. 0 => 0];
            end if;
            for I in Conversation'First .. Last_Msg loop
               if Conversation (I).Role = Role_Assistant then Last_Stable := I; end if;
            end loop;
            for I in Conversation'First .. Last_Stable loop
               declare
                  L : constant Natural :=
                    Conv_Ids (M, Conversation (Conversation'First .. I),
                              Conversation'First)'Length;
               begin
                  if L > 0 and then NB < Tmp'Last then
                     NB := NB + 1; Tmp (NB) := L;
                  end if;
               end;
            end loop;
            return Tmp (1 .. NB);
         end Hist_Bounds;
         Hist_B : constant Cache_Bounds_Array := Hist_Bounds;
         Ids : constant LLM_Tokenizer.Token_Array :=
           (if Think_Open then
              --  Empty-think prefill + thinking on: OPEN a <think> block so the
              --  model reasons visibly, exactly like Ollama's always-think
              --  template. The parser is seeded in reasoning (Think_Open) so the
              --  chain-of-thought streams as reasoning_content and the model's
              --  own </think> switches it to the answer.
              Prefix
              & One (M.Im_Start_Id)
              & LLM_Tokenizer.Encode (M.Tok, "assistant" & LF)
              & Encode_Chat (M, "<think>" & LF)
            elsif Last_Is_Asst then
              --  Real assistant prefill (non-empty, or empty-think with thinking
              --  off): open that turn and let the model continue it — NOT a
              --  closed turn + a second opener (that double turn was malformed).
              Prefix
              & One (M.Im_Start_Id)
              & LLM_Tokenizer.Encode (M.Tok, "assistant" & LF)
              & Encode_Chat
                  (M, To_String (Conversation (Conversation'Last).Text))
            else
              Prefix
              & One (M.Im_Start_Id)
              & (if Params_Eff.Enable_Thinking then
                    LLM_Tokenizer.Encode (M.Tok, "assistant" & LF)
                 else
                    LLM_Tokenizer.Encode (M.Tok, "assistant" & LF)
                    & Encode_Chat
                        (M, "<think>" & LF & LF & "</think>" & LF & LF)));
         --  Feed each token into the parser AS it is generated (Bridge.Emit ->
         --  Feed), so On_Text/On_Reasoning/On_Tool_Call fire live on SinkRef.
         --  A dead downstream client makes SinkRef raise, Decode_Tokens frees
         --  GPU state and re-raises, and generation stops within one token.
         Bridge : aliased Parser_Bridge :=
           (Token_Sink with Psr => P'Access, Tgt => SinkRef);
         Vis_Setup : constant Boolean := Setup_Vision (Ids);
         pragma Unreferenced (Vis_Setup);
         Raw : constant String :=
           Decode_Tokens (M, Ids, Max_New_Tokens, M.Im_End_Id, M.Eos_Id,
                          Bridge'Access, Params_Eff, Stats,
                          Prefix_Len => Sys_Prefix_Len,
                          Cache_Bounds => Hist_B,
                          Reason_Seeded => Think_Open);
         pragma Unreferenced (Raw);  -- already fed to the parser live above
      begin
         LLM_Qwen_GPU.Clear_Vision;
         LLM_Chat_Parser.Finalize (P, SinkRef);
         declare
            NC : constant Natural := LLM_Chat_Parser.N_Tool_Calls (P);
         begin
            return R : Chat_Result (NC) do
               R.Reasoning := To_Unbounded_String
                 (LLM_Chat_Parser.Reasoning_Of (P));
               R.Answer    := To_Unbounded_String
                 (LLM_Chat_Parser.Answer_Of (P));
               R.Finish    := To_Unbounded_String
                 (LLM_Chat_Parser.Finish_Of (P));
               declare
                  Parsed : constant LLM_Chat_Parser.Tool_Call_Array :=
                    LLM_Chat_Parser.Tool_Calls_Of (P);
               begin
                  for I in 1 .. NC loop
                     R.Tool_Calls (I) :=
                       (Id           => Parsed (I).Id,
                        Name         => Parsed (I).Name,
                        Arguments_JS => Parsed (I).Arguments_JS);
                  end loop;
               end;
            end return;
         end;
      end;
   end Chat_Raw;

   --  Non-streaming Chat: returns Chat_Result directly. No sink events.
   function Chat
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer := 256;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access Gen_Stats := null) return Chat_Result
   is
   begin
      return Chat_Raw (M, Conversation, Max_New_Tokens, null, Params, Stats);
   end Chat;

   --  Streaming Chat: Chat_Sink callbacks fire as the parser walks the
   --  model's raw output. Returns the same Chat_Result for convenience.
   function Chat
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access Chat_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access Gen_Stats := null) return Chat_Result is
   begin
      return Chat_Raw (M, Conversation, Max_New_Tokens, Sink, Params, Stats);
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
            if B.Is_MoE then
               C := C + LLM_Weight.Count (B.MoE.Gate_Inp_W)
                       + LLM_Weight.Count (B.MoE.Gate_Exp_W)
                       + LLM_Weight.Count (B.MoE.Up_W)
                       + LLM_Weight.Count (B.MoE.Down_W)
                       + LLM_Weight.Count (B.MoE.Shexp_Gate_W)
                       + LLM_Weight.Count (B.MoE.Shexp_Up_W)
                       + LLM_Weight.Count (B.MoE.Shexp_Down_W)
                       + Safe_N (B.MoE.Shexp_Gate_Inp_W);
            else
               C := C + LLM_Weight.Count (B.Dense.Gate_W)
                       + LLM_Weight.Count (B.Dense.Up_W)
                       + LLM_Weight.Count (B.Dense.Down_W);
            end if;
         end;
      end loop;
      return C;
   end Param_Count;

   function Vocab_Size  (M : Qwen_Model) return Integer is (M.Vocab_Sz);
   function Context_Len (M : Qwen_Model) return Integer is (M.Ctx_Len);
   function Dim         (M : Qwen_Model) return Integer is (M.Model_Dim);
   function Block_Count (M : Qwen_Model) return Integer is (M.N_Blocks);
   function Arch_Name   (M : Qwen_Model) return String is
     (To_String (M.Arch));

   function Encode (M : Qwen_Model; Text : String)
     return LLM_Tokenizer.Token_Array is
     (LLM_Tokenizer.Encode (M.Tok, Text));

   function Decode (M : Qwen_Model; Ids : LLM_Tokenizer.Token_Array)
     return String is
     (LLM_Tokenizer.Decode (M.Tok, Ids));

end LLM_Qwen;
