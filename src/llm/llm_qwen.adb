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
   --  Phase C: full resident forward chain — registered once per loaded
   --  model (keyed by the embedding table address). When registered AND all
   --  per-generation states allocated on the device, Decode runs one
   --  Chain_Forward per token: embedding row in, logits out, hidden state
   --  never leaves VRAM.
   Chain_Tag : System.Address := System.Null_Address;
   Chain_OK  : Boolean := False;
   Bench_Done : Boolean := False;

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

   function Decode_Tokens
     (M              : Qwen_Model;
      Prompt_Ids     : LLM_Tokenizer.Token_Array;
      Max_New_Tokens : Integer;
      Stop1, Stop2   : Integer;
      Sink           : access Token_Sink'Class;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access Gen_Stats := null) return String
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

      --  Per-layer decode state (KV cache for full-attn, recurrent state +
      --  conv window for delta-net), threaded across tokens. One forward
      --  step costs O(1) matmuls instead of recomputing the sequence.
      Cache : array (1 .. M.N_Blocks) of LLM_Qwen_Blk.Block_State;

      --  Phase C chain: per-generation state handles + token position.
      Use_Chain : Boolean := False;
      Chain_Pos : Natural := 0;
      Handles   : array (1 .. M.N_Blocks) of Interfaces.C.int :=
        [others => Interfaces.C.int (Integer'(-1))];

      procedure Free_States is
      begin
         LLM_Qwen_GPU.Chain_End;
         for I in Cache'Range loop
            if Cache (I).Is_Full then
               LLM_Qwen_GPU.Fattn_Free (Cache (I).Full_St.GPU_Handle);
               Cache (I).Full_St.GPU_Handle := -1;
            else
               LLM_Qwen_GPU.Dnet_Free (Cache (I).DNet_St.GPU_Handle);
               Cache (I).DNet_St.GPU_Handle := -1;
            end if;
         end loop;
      end Free_States;

      --  One forward step under the shared step lock, released between steps
      --  (incl. on exception) so concurrent generations interleave per token.
      function Decode (Embed_Row : Integer) return Tensor is
         use type Ada.Real_Time.Time;
         H  : Tensor := New_Tensor ([1, Dim]);
         TS : Ada.Real_Time.Time;
      begin
         LLM_Step_Lock.Acquire;
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
               LLM_Step_Lock.Release;
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
            LLM_Step_Lock.Release;
            return R;
         end;
      exception
         when others =>
            LLM_Step_Lock.Release;
            raise;
      end Decode;

      Last_Logits : Tensor;
   begin
      for I in 1 .. M.N_Blocks loop
         Cache (I) := LLM_Qwen_Blk.Init_State (M.Blocks (I).all, Cap);
      end loop;

      --  Phase C: use the resident chain when the model is registered and
      --  every layer's device state was allocated.
      Register_Chain (M);
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
      if Use_Chain then
         LLM_Qwen_GPU.Chain_Begin (Handles (1)'Address);
      end if;

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
            Tid : constant Integer := LLM_Sampler.Next
              (Smp, Last_Logits, Hist (N_Hist - Win + 1 .. N_Hist));
            Best_Row : constant Integer := Tid + 1;   -- 1-based embedding row
         begin
            exit when Best_Row < 1 or else Best_Row > M.Vocab_Sz;
            if Tid = Stop1 or else Tid = Stop2 then         -- natural stop
               Hit_Stop := True;
               exit;
            end if;
            declare
               Piece : constant String := LLM_Tokenizer.Decode_One (M.Tok, Tid);
            begin
               Append (Out_Buf, Piece);
               if Sink /= null then            -- stream this token now
                  Sink.Emit (Piece);
               end if;
            end;
            Produced := Produced + 1;
            N_Hist := N_Hist + 1; Hist (N_Hist) := Tid;
            exit when Step = Max_New_Tokens;
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

   --  Structured chat internal: build the prompt, generate raw pieces,
   --  drive the FSM parser, and return the assembled Chat_Result. Used by
   --  both the streaming (with Chat_Sink) and non-streaming variants.
   function Chat_Raw
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer;
      Sink : access Chat_Sink'Class;
      Params : LLM_Sampler.Params;
      Stats : access Gen_Stats) return Chat_Result
   is
      LF     : constant String := [1 => ASCII.LF];
      P      : LLM_Chat_Parser.Parser := LLM_Chat_Parser.New_Parser;
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
         Ids : constant LLM_Tokenizer.Token_Array :=
           Conv_Ids (M, Conversation, Conversation'First)
           & One (M.Im_Start_Id)
           & LLM_Tokenizer.Encode (M.Tok, "assistant" & LF);
         Raw : constant String :=
           Decode_Tokens (M, Ids, Max_New_Tokens, M.Im_End_Id, M.Eos_Id,
                          null, Params, Stats);
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

end LLM_Qwen;
