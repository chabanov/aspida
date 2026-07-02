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
with LLM_Dequant;
with LLM_GPU;
with LLM_Pool;
with LLM_Step_Lock;
with Ctx_Window;

package body LLM_Llama is

   use Ada.Strings.Fixed;

   --  Strict mode: refuse an over-window request with context_length_exceeded
   --  (OpenAI semantics) instead of the default graceful turn-trim.
   Strict_Ctx : constant Boolean :=
     Ada.Environment_Variables.Exists ("ASPIDA_STRICT_CTX");

   --  Context-shift ("infinite" generation): on by default (like llama.cpp).
   --  When a slot fills its window mid-generation, evict the oldest KV and keep
   --  going instead of stopping. Disable with ASPIDA_NO_CTX_SHIFT.
   Ctx_Shift_On : constant Boolean :=
     not Ada.Environment_Variables.Exists ("ASPIDA_NO_CTX_SHIFT");
   N_Sink : constant := 4;   -- attention-sink tokens kept at the front

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

   ------------------------------------------------------------------
   --  H19 Phase 7 partial-warm: per-block "is this layer in RAM yet"
   --  barrier + the background streamer that fills blocks K+1..N while
   --  inference runs on the first K. See Llama_Model_Rec below for the
   --  race-free protocol (Sched reads only after Wait_Ready; the fetcher
   --  writes only before Mark_Ready; the fetcher never touches
   --  LLM_Step_Lock).
   ------------------------------------------------------------------
   --  STICKY gate + one-shot wake per layer. The fetcher loads layer I,
   --  flips the sticky Loaded (I) flag, then Set_True's the one-shot Wake
   --  (I) object to release any forward pass blocked on it. The forward
   --  pass takes the sticky fast path (Loaded (I) => return) once a layer
   --  is confirmed in RAM, and only Suspend_Until_True's the wake on the
   --  first visit — Suspend_Until_True is called OUTSIDE any protected
   --  lock (it would self-deadlock inside one), so Warm_Barrier is a plain
   --  limited record. The fetcher signals a mid-stream break by setting
   --  Failed and Set_True'ing every layer's Wake, so any waiter wakes and
   --  re-checks Failed -> Fetch_Error.
   --
   --  WHY sticky + one-shot instead of a single Suspension_Object per
   --  layer: Suspend_Until_True RESETS its object to False on return (it
   --  is a one-shot signal, not a sticky flag — verified empirically: a
   --  second Suspend on the same object blocks even after Set_True). A
   --  layer being in RAM is a permanent condition, so the gate must be
   --  sticky; the wake object is the consumable signal. An earlier draft
   --  used a protected entry family indexed by Positive, but GNAT
   --  materializes per-family-index state and a 1 .. Positive'Last family
   --  elaborates forever; a bounded family would work but imposes an
   --  artificial layer cap. This design has no cap and reuses the STC
   --  idiom already used by the scheduler's slot-completion handshake.
   type Bool_Array is array (Positive range <>) of Boolean;
   type SO_Array is array (Positive range <>)
     of Ada.Synchronous_Task_Control.Suspension_Object;
   type Warm_Barrier (N : Positive) is limited record
      Loaded : Bool_Array (1 .. N) := [others => False];  --  sticky: layer in RAM
      Wake   : SO_Array (1 .. N);                          --  one-shot wake signal
      Failed : Boolean := False;
      pragma Atomic (Failed);             --  fetcher writes, forward reads
   end record;
   type Warm_Ptr is access Warm_Barrier;

   --  Fetcher: layer I is in RAM (set the sticky flag, then wake waiters).
   procedure Mark_Ready (B : in out Warm_Barrier; I : Positive);
   --  Fetcher: the stream broke; wake every waiter so it raises Fetch_Error
   --  instead of blocking on a layer that will never arrive.
   procedure Fail_Barrier (B : in out Warm_Barrier);
   --  Forward pass: block until layer I is in RAM; raise Fetch_Error if the
   --  fetcher failed before I landed. Sticky fast path once loaded.
   procedure Wait_Ready (B : in out Warm_Barrier; I : Positive);

   --  Re-export the shared heap-GGUF access type under a short local name.
   subtype GGUF_Ptr is LLM_GGUF.GGUF_Ptr;

   task type Block_Fetcher is
      entry Init (Mdl : Llama_Model; K_Warm : Positive);
      entry Shutdown;
   end Block_Fetcher;
   type Fetch_Ptr is access Block_Fetcher;

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
      --  H19 Phase 7 partial-warm state. All null/false/0 on the eager path
      --  (Warm = null => the forward pass skips the per-block Wait entirely).
      Warm         : Warm_Ptr  := null;  --  per-block ready barrier
      GGUF         : GGUF_Ptr  := null;  --  kept-alive source for the fetcher
      Fetcher      : Fetch_Ptr := null;  --  background block streamer
      Warm_Count   : Natural   := 0;     --  K (layers loaded eagerly)
      Fetch_Failed : Boolean   := False; --  fetcher hit a transport error
   end record;

   --  Deallocators (declared early so the Block_Fetcher body, the partial-load
   --  failure handler, and Free can all reference them).
   procedure Free_Rec is
     new Ada.Unchecked_Deallocation (Llama_Model_Rec, Llama_Model);
   procedure Free_Blocks is
     new Ada.Unchecked_Deallocation (Block_Arr, Block_Arr_Ptr);
   procedure Free_Warm is
     new Ada.Unchecked_Deallocation (Warm_Barrier, Warm_Ptr);
   procedure Free_GGUF is
     new Ada.Unchecked_Deallocation (LLM_GGUF.GGUF_File, GGUF_Ptr);
   procedure Free_Fetcher is
     new Ada.Unchecked_Deallocation (Block_Fetcher, Fetch_Ptr);

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
   -- H19 Phase 7: shared tensor readers + per-block loader.
   --
   --  Factored out of Load_From_File's nested LQ/LT so the background
   --  Block_Fetcher can load a layer with the SAME code path as the eager
   --  load (no duplicated dequant logic to drift). Load_Weight reads a
   --  named tensor's raw quantized bytes from G and builds a Weight;
   --  Load_Tensor additionally dequantizes row 0 into a Tensor (matching the
   --  existing LT behaviour — the Weight's quant bytes are left for the
   --  caller's Free path, exactly as the eager path did). Load_Block reads
   --  all 9 tensors of one transformer block.
   --------------------------------------------------------------------

   procedure Load_Weight
     (G    : in out GGUF_File;
      Name : String;
      W    : out LLM_Weight.Weight)
   is
      Info : constant Tensor_Info := Find_Tensor (G, Name);
      Size : constant Natural := Natural (Tensor_Byte_Size (Info));
      B    : LLM_Weight.Byte_Data;
   begin
      --  Reject an unimplemented quantization up front (else a quantized
      --  matrix would only fail at first inference, or worse, run garbage).
      if not LLM_Dequant.Is_Supported (Info.Kind) then
         raise Model_Load_Error with "weight " & Name
           & ": unsupported quantization "
           & LLM_GGUF.GGML_Type'Image (Info.Kind);
      end if;
      B := new String (1 .. Size);
      Read_Tensor_Raw (G, Info, B.all'Address, Size);
      W := LLM_Weight.From_Quant (Info, B);
   exception
      when E : others =>
         raise Model_Load_Error with "weight " & Name & ": "
           & Ada.Exceptions.Exception_Message (E);
   end Load_Weight;

   procedure Load_Tensor
     (G    : in out GGUF_File;
      Name : String;
      T    : out Tensor)
   is
      W : LLM_Weight.Weight;
   begin
      Load_Weight (G, Name, W);
      T := LLM_Weight.Get_Row (W, 0);   --  dequant row 0 (matches old LT)
   end Load_Tensor;

   procedure Load_Block
     (G      : in out GGUF_File;
      Prefix : String;
      Bk     : out L_Block)
   is
   begin
      Load_Tensor (G, Prefix & "attn_norm.weight", Bk.Attn_Norm);
      Load_Tensor (G, Prefix & "ffn_norm.weight",  Bk.Ffn_Norm);
      Load_Weight (G, Prefix & "attn_q.weight",      Bk.W_Q);
      Load_Weight (G, Prefix & "attn_k.weight",      Bk.W_K);
      Load_Weight (G, Prefix & "attn_v.weight",      Bk.W_V);
      Load_Weight (G, Prefix & "attn_output.weight", Bk.W_O);
      Load_Weight (G, Prefix & "ffn_gate.weight",    Bk.W_Gate);
      Load_Weight (G, Prefix & "ffn_up.weight",      Bk.W_Up);
      Load_Weight (G, Prefix & "ffn_down.weight",    Bk.W_Down);
   end Load_Block;

   --------------------------------------------------------------------
   --  Warm_Barrier primitives: a sticky "loaded" flag + a one-shot wake
   --  per layer, plus a fail flag. Set_True / Suspend_Until_True carry
   --  their own acquire/release fence, so a Loaded/Failed write performed
   --  before Set_True is visible to a task woken by Suspend_Until_True.
   --  The sticky fast path reads Loaded without a fence, which is safe:
   --  a layer is only ever Loaded False -> True (once), and the first
   --  visit that wakes via Suspend carries the fence that publishes it.
   --------------------------------------------------------------------
   procedure Mark_Ready (B : in out Warm_Barrier; I : Positive) is
   begin
      B.Loaded (I) := True;   --  sticky: layer I is in RAM from here on
      Ada.Synchronous_Task_Control.Set_True (B.Wake (I));
   end Mark_Ready;

   procedure Fail_Barrier (B : in out Warm_Barrier) is
   begin
      B.Failed := True;
      for I in 1 .. B.N loop
         Ada.Synchronous_Task_Control.Set_True (B.Wake (I));
         --  wake every waiter; each re-checks Failed and raises Fetch_Error
      end loop;
   end Fail_Barrier;

   procedure Wait_Ready (B : in out Warm_Barrier; I : Positive) is
   begin
      if B.Loaded (I) then
         return;   --  sticky fast path: layer already in RAM
      end if;
      Ada.Synchronous_Task_Control.Suspend_Until_True (B.Wake (I));
      if B.Failed then
         raise Fetch_Error;
      end if;
      pragma Assert (B.Loaded (I));   --  fetcher sets Loaded before Set_True
   end Wait_Ready;

   --------------------------------------------------------------------
   -- Load
   --------------------------------------------------------------------

   function Load (Path : String) return Llama_Model is
      G : GGUF_File;
      M : Llama_Model;
   begin
      Open (G, Path);
      if not Is_Open (G) then
         raise Model_Load_Error with "cannot open GGUF file: " & Path;
      end if;
      --  Load_From_File reads the tensors and closes G (freeing the source).
      Load_From_File (G, M);
      return M;
   end Load;

   ---------------------------------------------------------------------
   -- Load_From_File — the tensor-reading core, factored out of Load so the
   -- H19 weight-streaming path can feed an already-open GGUF_File (one whose
   -- byte source is a Remote_AEAD_Source). See spec.
   ---------------------------------------------------------------------
   procedure Load_From_File (G : in out GGUF_File; M : out Llama_Model) is

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
         --  Thin wrapper over the shared Load_Weight so the eager head load
         --  and the Phase 7 fetcher use one dequant code path.
         W : LLM_Weight.Weight;
      begin
         Load_Weight (G, Name, W);
         return W;
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
      M := new Llama_Model_Rec;
      Ada.Text_IO.Put_Line ("Loading Llama (dense) model from open GGUF source ...");
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
            Bk : L_Block;
         begin
            Load_Block (G, "blk." & Img (I - 1) & ".", Bk);
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
      --  Long-context RoPE scaling. We implement the well-understood LINEAR
      --  (Position Interpolation) method; "yarn" is read but not yet applied
      --  (its per-dim ramp needs reference validation), so we leave such models
      --  unscaled rather than risk wrong math.
      declare
         Kind   : constant String := Metadata (G, "llama.rope.scaling.type");
         Factor : constant Float  := MF ("rope.scaling.factor", 1.0);
      begin
         if Kind = "linear" and then Factor > 1.0 then
            LLM_RoPE.Set_Linear_Scale (M.RoPE, Factor);
            Ada.Text_IO.Put_Line ("  rope scaling: linear x" & Factor'Image);
         elsif Kind = "yarn" and then Factor > 1.0 then
            --  Full YaRN (llama.cpp rope_yarn): per-dim ramp + attention temp.
            LLM_RoPE.Set_Yarn_Scale
              (M.RoPE, Factor,
               N_Ctx_Orig => MI ("rope.scaling.original_context_length",
                                 M.Ctx / Integer'Max (1, Integer (Factor))));
            Ada.Text_IO.Put_Line ("  rope scaling: yarn x" & Factor'Image);
         end if;
         --  Operator override: extend an unscaled model's context (NTK-aware).
         if Ada.Environment_Variables.Exists ("ASPIDA_ROPE_NTK") then
            begin
               LLM_RoPE.Set_NTK_Scale
                 (M.RoPE,
                  Float'Value (Ada.Environment_Variables.Value ("ASPIDA_ROPE_NTK")));
               Ada.Text_IO.Put_Line ("  rope scaling: NTK override applied");
            exception when others => null;
            end;
         end if;
      end;

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
   exception
      --  On any mid-load failure, ensure G (and its byte source) is freed —
      --  the success path closes G above; a partial load must not orphan it.
      when others =>
         if Is_Open (G) then
            Close (G);
         end if;
         raise;
   end Load_From_File;

   ---------------------------------------------------------------------
   -- Load_From_File_Partial — H19 Phase 7 partial-model warm.
   --
   --  Loads the head + the first K transformer blocks EAGERLY, then (if
   --  K < block_count) hands the still-open GGUF source to a background
   --  Block_Fetcher that streams blocks K+1..N into RAM in ascending order
   --  while inference runs on the first K. The forward pass blocks per-layer
   --  via M.Warm.Wait only if it out-runs the fetcher (race-free: the fetcher
   --  writes M.Blocks(I) BEFORE M.Warm.Mark(I); the forward pass reads
   --  M.Blocks(I) only AFTER M.Warm.Wait(I) returns; protected-object
   --  rendezvous supplies the happens-before edge).
   --
   --  Takes ownership of the heap-allocated G: the model keeps it alive
   --  (M.GGUF) for the fetcher, which closes + frees it when all blocks are
   --  in RAM (or on a fetch failure). On a mid-load failure G is closed +
   --  freed here and Model_Load_Error is raised. K is clamped to 1..
   --  block_count; K >= block_count degenerates to the eager full load (all
   --  blocks read here, source closed, no fetcher started).
   ---------------------------------------------------------------------
   procedure Load_From_File_Partial
     (G : GGUF_Ptr; M : out Llama_Model; K : Positive)
   is

      function MI (Key : String; D : Integer) return Integer is
         V : constant String := Metadata (G.all, "llama." & Key);
      begin
         return (if V = "" then D else Integer'Value (V));
      exception when others => return D; end MI;

      function MF (Key : String; D : Float) return Float is
         V : constant String := Metadata (G.all, "llama." & Key);
      begin
         return (if V = "" then D else Float'Value (V));
      exception when others => return D; end MF;

      function LQ (Name : String) return LLM_Weight.Weight is
         W : LLM_Weight.Weight;
      begin
         Load_Weight (G.all, Name, W);
         return W;
      end LQ;

      function Has (Name : String) return Boolean is
      begin
         declare Unused : constant Tensor_Info := Find_Tensor (G.all, Name); begin
            pragma Unreferenced (Unused);
            return True;
         end;
      exception when others => return False;
      end Has;

      function LT (Name : String) return Tensor is (LLM_Weight.Get_Row (LQ (Name), 0));

      RoPE_Base : Float;
      K_Eff     : Integer;   --  clamped warm layer count
   begin
      M := new Llama_Model_Rec;
      --  Take ownership of the heap GGUF source at entry so the failure
      --  handler can always close + free it uniformly.
      M.GGUF := G;

      Ada.Text_IO.Put_Line
        ("Loading Llama (dense) model from open GGUF source (partial warm) ...");
      if Metadata (G.all, "general.architecture") /= "llama" then
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
      M.Output   := (if Has ("output.weight") then LQ ("output.weight") else M.Tok_Emb);
      M.Vocab    := LLM_Weight.Rows (M.Output);
      M.Has_Freqs := Has ("rope_freqs.weight");
      if M.Has_Freqs then M.Rope_Freqs := LT ("rope_freqs.weight"); end if;

      --  Clamp K to a valid warm range. K_Eff >= block_count => degenerate
      --  eager full load (handled below: all blocks read here, no fetcher).
      K_Eff := Integer'Max (1, Integer'Min (K, M.N_Blocks));

      M.Blocks := new Block_Arr (1 .. M.N_Blocks);
      if K_Eff < M.N_Blocks then
         M.Warm := new Warm_Barrier (M.N_Blocks);
      end if;

      --  Eagerly load the first K_Eff blocks; mark each ready so a forward
      --  pass starting at once never waits on these.
      for I in 1 .. K_Eff loop
         declare
            Bk : L_Block;
         begin
            Load_Block (G.all, "blk." & Img (I - 1) & ".", Bk);
            M.Blocks (I) := Bk;
         end;
         if M.Warm /= null then
            Mark_Ready (M.Warm.all, I);
         end if;
      end loop;

      M.Head_Dim := LLM_Weight.Rows (M.Blocks (1).W_Q) / M.N_Heads;
      M.RoPE := LLM_RoPE.Create_Qwen_RoPE (M.Head_Dim, RoPE_Base, M.Ctx);
      LLM_RoPE.Set_Interleaved (M.RoPE);
      if M.Has_Freqs and then not Ada.Environment_Variables.Exists ("ASPIDA_NO_FF") then
         LLM_RoPE.Set_Freq_Factors (M.RoPE, M.Rope_Freqs);
      end if;
      declare
         Kind   : constant String := Metadata (G.all, "llama.rope.scaling.type");
         Factor : constant Float  := MF ("rope.scaling.factor", 1.0);
      begin
         if Kind = "linear" and then Factor > 1.0 then
            LLM_RoPE.Set_Linear_Scale (M.RoPE, Factor);
         elsif Kind = "yarn" and then Factor > 1.0 then
            LLM_RoPE.Set_Yarn_Scale
              (M.RoPE, Factor,
               N_Ctx_Orig => MI ("rope.scaling.original_context_length",
                                 M.Ctx / Integer'Max (1, Integer (Factor))));
         end if;
         if Ada.Environment_Variables.Exists ("ASPIDA_ROPE_NTK") then
            begin
               LLM_RoPE.Set_NTK_Scale
                 (M.RoPE,
                  Float'Value (Ada.Environment_Variables.Value ("ASPIDA_ROPE_NTK")));
            exception when others => null;
            end;
         end if;
      end;

      M.Tok := LLM_Tokenizer.Create;
      LLM_Tokenizer.Load_From_GGUF (M.Tok, G.all);
      begin M.Bos := Integer'Value (Metadata (G.all, "tokenizer.ggml.bos_token_id"));
      exception when others => M.Bos := LLM_Tokenizer.Token_To_Id (M.Tok, "<|begin_of_text|>"); end;
      begin M.Eos := Integer'Value (Metadata (G.all, "tokenizer.ggml.eos_token_id"));
      exception when others => M.Eos := LLM_Tokenizer.Token_To_Id (M.Tok, "<|end_of_text|>"); end;
      M.Eot := LLM_Tokenizer.Token_To_Id (M.Tok, "<|eot_id|>");
      M.SH  := LLM_Tokenizer.Token_To_Id (M.Tok, "<|start_header_id|>");
      M.EH  := LLM_Tokenizer.Token_To_Id (M.Tok, "<|end_header_id|>");

      --  Hand the still-open source to a background fetcher for the remaining
      --  blocks, OR (degenerate) close it now and serve a fully-loaded model.
      if K_Eff < M.N_Blocks then
         M.Warm_Count := K_Eff;
         M.Fetcher := new Block_Fetcher;
         M.Fetcher.Init (M, K_Eff);
         Ada.Text_IO.Put_Line
           ("  llama partial-warm: " & Img (K_Eff) & "/" & Img (M.N_Blocks)
            & " layers hot; streaming the rest in the background");
      else
         --  Degenerate eager full load: free the source now (the record is
         --  freed by Free later; no fetcher reads it).
         Close (M.GGUF.all);
         Ada.Text_IO.Put_Line ("  llama partial-warm: K >= block_count, full load");
      end if;

      Ada.Text_IO.Put_Line ("  llama: dim=" & Img (M.Dim)
        & " layers=" & Img (M.N_Blocks) & " heads=" & Img (M.N_Heads)
        & "/" & Img (M.N_KV) & " head_dim=" & Img (M.Head_Dim)
        & " ffn=" & Img (M.FFN) & " vocab=" & Img (M.Vocab)
        & (if M.Has_Freqs then " rope_freqs" else ""));
   exception
      --  Fail-loud, no source leak. The fetcher is the LAST thing started, so
      --  on any mid-load failure it isn't running and we may safely close +
      --  free M.GGUF ourselves. The partially-built model record (some blocks,
      --  head weights, Warm) is leaked, matching Load_From_File's failure
      --  semantics (load failure is fatal; the caller gets Model_Load_Error).
      when others =>
         if M /= null then
            if M.Fetcher /= null then
               begin M.Fetcher.Shutdown; exception when others => null; end;
               while not M.Fetcher'Terminated loop delay 0.001; end loop;
               Free_Fetcher (M.Fetcher);
            end if;
            if M.GGUF /= null then
               if Is_Open (M.GGUF.all) then Close (M.GGUF.all); end if;
               Free_GGUF (M.GGUF);
               M.GGUF := null;
            end if;
            if M.Warm /= null then Free_Warm (M.Warm); end if;
         end if;
         raise;
   end Load_From_File_Partial;

   ---------------------------------------------------------------------
   -- Block_Fetcher task body — streams blocks K+1..N in the background.
   --  Marks each ready as it lands; on a transport/dequant error fails the
   --  barrier (so a forward pass waiting on a not-yet-loaded layer raises
   --  Fetch_Error instead of hanging). Closes + frees M.GGUF on the way out
   --  (both success and failure), then parks on Shutdown for Free's
   --  deterministic rendezvous.
   ---------------------------------------------------------------------
   task body Block_Fetcher is
      --  Local copies of the Init rendezvous values. The accept's formals
      --  MUST match the entry declaration's names (Mdl, K_Warm) and are `in`,
      --  so we copy them into differently-named locals to use below.
      M_Model : Llama_Model;
      M_K     : Positive;
   begin
      accept Init (Mdl : Llama_Model; K_Warm : Positive) do
         M_Model := Mdl;
         M_K     := K_Warm;
      end Init;

      for I in M_K + 1 .. M_Model.N_Blocks loop
         begin
            declare
               Bk : L_Block;
            begin
               Load_Block (M_Model.GGUF.all, "blk." & Img (I - 1) & ".", Bk);
               M_Model.Blocks (I) := Bk;
            end;
            Mark_Ready (M_Model.Warm.all, I);
         exception
            when others =>
               --  Stream broke mid-fetch. Fail-loud: wake every waiter so the
               --  forward pass raises Fetch_Error rather than hanging on a
               --  layer that will never arrive. Leave remaining blocks empty.
               M_Model.Fetch_Failed := True;
               Fail_Barrier (M_Model.Warm.all);
               exit;
         end;
      end loop;

      --  Sole reader of M_Model.GGUF is done; close the source and drop our
      --  handle so Free's GGUF guard becomes a no-op (Free checks M.GGUF).
      if M_Model.GGUF /= null then
         if Is_Open (M_Model.GGUF.all) then
            Close (M_Model.GGUF.all);
         end if;
         Free_GGUF (M_Model.GGUF);
         M_Model.GGUF := null;
      end if;

      --  Park until Free calls Shutdown — gives Free a deterministic
      --  "fetcher quiescent" handshake before it frees M.Blocks / the rec.
      accept Shutdown;
   end Block_Fetcher;

   --------------------------------------------------------------------
   --  Free — full teardown for Phase 1b LRU eviction.
   --------------------------------------------------------------------
   --  Drop a weight's GPU mirror (if any) THEN its host bytes, in that order:
   --  the host address is the device cache key and must be valid for the free.
   procedure Drop_Weight (W : in out LLM_Weight.Weight) is
   begin
      LLM_GPU.Free_Weight (LLM_Weight.Raw_Address (W));
      LLM_Weight.Free_Bytes (W);
   end Drop_Weight;

   procedure Free (M : in out Llama_Model) is
   begin
      if M = null then
         return;   --  idempotent
      end if;

      --  H19 Phase 7 teardown: stop the background fetcher first. It may be
      --  mid-Read_Tensor_Raw on M.GGUF — we must NOT free M.GGUF / M.Blocks
      --  out from under it. Rendezvous: call Shutdown (the fetcher's last
      --  accept), then wait for the task to actually terminate before freeing
      --  anything it could touch. Wrapped: a fetcher that already finished its
      --  loop and is blocked on accept Shutdown returns normally; one that died
      --  on a fetch error has already set M.Fetch_Failed and may reject the
      --  rendezvous — the handler lets us proceed to the terminated-poll.
      if M.Fetcher /= null then
         begin
            M.Fetcher.Shutdown;
         exception
            when others => null;
         end;
         --  The fetcher task closes + nulls M.GGUF itself on its way out (both
         --  on the success path and the Fail path), so by the time it is
         --  Terminated M.GGUF is null. Poll rather than abort: aborting a task
         --  mid-Read_Tensor_Raw could leave the channel in a half-read state.
         while not M.Fetcher'Terminated loop
            delay 0.001;
         end loop;
         Free_Fetcher (M.Fetcher);
      end if;

      --  The fetcher normally closes + nulls M.GGUF. Guard anyway: if Free is
      --  called on a model whose fetcher never started (Warm = null eager
      --  degenerate path that still set M.GGUF), or the fetcher crashed before
      --  its close, M.GGUF may still hold an open source — close + free it.
      if M.GGUF /= null then
         if Is_Open (M.GGUF.all) then
            Close (M.GGUF.all);
         end if;
         Free_GGUF (M.GGUF);
      end if;

      if M.Warm /= null then
         Free_Warm (M.Warm);
      end if;

      if M.Blocks /= null then
         for I in M.Blocks'Range loop
            declare
               B : L_Block renames M.Blocks (I);
            begin
               Drop_Weight (B.W_Q);    Drop_Weight (B.W_K);
               Drop_Weight (B.W_V);    Drop_Weight (B.W_O);
               Drop_Weight (B.W_Gate); Drop_Weight (B.W_Up);
               Drop_Weight (B.W_Down);
               --  Attn_Norm / Ffn_Norm are controlled Tensors: they finalize
               --  when the block array is deallocated below.
            end;
         end loop;
         Free_Blocks (M.Blocks);   --  finalizes the per-block norm Tensors
      end if;

      Drop_Weight (M.Tok_Emb);
      Drop_Weight (M.Output);
      --  Out_Norm / Rope_Freqs are controlled Tensors finalized with the record.

      Free_Rec (M);   --  nulls M (idempotent on a second call)
   end Free;

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
         --  H19 Phase 7 partial-warm: block until layer Lr is in RAM. On the
         --  eager path M.Warm is null and this is a no-op. On the partial path
         --  the declarative part below reads M.Blocks (Lr) (via the renames +
         --  the X initializer), so the Wait MUST precede it. Raises
         --  Fetch_Error if the background streamer failed before Lr landed.
         if M.Warm /= null then
            Wait_Ready (M.Warm.all, Lr);
         end if;
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
         --  H19 Phase 7 partial-warm: same per-block gate as Forward_Step.
         if M.Warm /= null then
            Wait_Ready (M.Warm.all, Lr);
         end if;
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
   --  Absolute ceiling for per-slot STATIC arrays (e.g. the sampler history).
   --  Cheap (a few KB/slot); the costly KV cache is sized to Ctx_Cap below.
   Sched_Cap_Max : constant := 32768;
   --  Effective context window per slot (prompt + generation), set once at
   --  scheduler init to min(model context, configured budget, ceiling). Default
   --  preserves the previous 4096 footprint; raise via ASPIDA_CTX up to the
   --  model's trained context. Read-only after init (single-shot Sched.Init).
   Ctx_Cap : Natural := 4096;

   --  Resolve the served window from the model's trained context and the
   --  ASPIDA_CTX budget, clamped to [256, Sched_Cap_Max]. Sizing to the real
   --  context (rather than a blind 4096) is honest and llama.cpp-like.
   function Configured_Ctx (Model_Ctx : Integer) return Natural is
      Mctx : constant Integer := Integer'Max (1, Model_Ctx);
      Want : Integer := 4096;   -- memory-preserving default budget
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_CTX") then
         Want := Integer'Value (Ada.Environment_Variables.Value ("ASPIDA_CTX"));
      end if;
      --  Effective window = min(budget, model context, static ceiling); NEVER
      --  exceeds the model's trained context (so a small-context model is
      --  served honestly, not over-promised).
      return Natural
        (Integer'Max (1,
           Integer'Min (Integer'Min (Integer'Max (1, Want), Mctx),
                        Sched_Cap_Max)));
   exception
      when others =>
         return Natural (Integer'Max (1, Integer'Min (Mctx, 4096)));
   end Configured_Ctx;

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
      --  Generation accounting (filled by the scheduler before Retire).
      Prompt_Toks : Natural := 0;
      Comp_Toks   : Natural := 0;
      Trunc       : Boolean := False;   -- retired on the cap, not a stop token
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
         Hist   : LLM_Sampler.History (1 .. Sched_Cap_Max) := [others => 0];
         N_Hist : Natural := 0;
      end record;
      Slots  : array (1 .. Sched_Max_Seq) of SSlot;
      Active : Natural := 0;

      function New_Cache return KV_Cache_Ptr is
         C : constant KV_Cache_Ptr := new KV_Cache (1 .. M.N_Blocks);
      begin
         for L in 1 .. M.N_Blocks loop
            C (L).K := new Tensor_Array (1 .. Ctx_Cap);
            C (L).V := new Tensor_Array (1 .. Ctx_Cap);
         end loop;
         return C;
      end New_Cache;

      --  Context-shift: free room in slot S's full KV cache. Keep N_Sink
      --  attention-sink tokens at the front + the most recent window, evict the
      --  oldest E middle positions, slide the retained K/V down and re-rotate
      --  the slid K back by E positions (RoPE composes additively, so the delta
      --  rotation is Apply at -E; V is not rotated). Returns positions freed.
      function Shift_KV (S : Integer) return Natural is
         Pos : constant Integer := Slots (S).Pos;       -- cached token count
         HD  : constant Integer := M.Head_Dim;
         NKV : constant Integer := M.N_KV;
         E   : constant Integer := Integer'Max (1, (Pos - N_Sink) / 2);
      begin
         if Pos <= N_Sink + E then return 0; end if;    -- nothing useful to free
         for L in 1 .. M.N_Blocks loop
            declare
               KC : Tensor_Array_Ptr renames Slots (S).Cache (L).K;
               VC : Tensor_Array_Ptr renames Slots (S).Cache (L).V;
            begin
               --  0-based positions: move [N_Sink+E .. Pos-1] down to [N_Sink ..].
               for P in N_Sink .. Pos - E - 1 loop
                  declare
                     Src     : constant Tensor := KC (P + E + 1);
                     Shifted : Tensor := New_Tensor ([1, NKV * HD]);
                  begin
                     for Hh in 0 .. NKV - 1 loop
                        declare
                           R : constant Tensor := LLM_RoPE.Apply
                             (M.RoPE, Slice (Src, Hh * HD + 1, HD), -E);
                        begin
                           for J in 1 .. HD loop
                              Set_Flat (Shifted, Hh * HD + J, Get_Flat (R, J));
                           end loop;
                        end;
                     end loop;
                     KC (P + 1) := Shifted;
                     VC (P + 1) := VC (P + E + 1);   -- V: slide only (not rotated)
                  end;
               end loop;
            end;
         end loop;
         Slots (S).Pos := Pos - E;
         Ada.Text_IO.Put_Line
           ("  [ctx-shift] window full; evicted" & E'Image
            & " oldest positions, kept" & N_Sink'Image & " sinks");
         Ada.Text_IO.Flush;
         return E;
      end Shift_KV;

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
      --  Wait for the (single-shot) Init, but allow this library-level task to
      --  terminate when the program ends without ever using the scheduler — the
      --  Forward_Logits / direct-forward tools never call Sched.Init, so without
      --  the terminate alternative this task would park here forever and block
      --  the whole program from exiting (collective termination needs every
      --  library task at a terminate point). The server always calls Init, so
      --  it takes the accept and runs the loop below indefinitely as before.
      select
         accept Init (Mdl : Llama_Model) do M := Mdl; end Init;
      or
         terminate;
      end select;
      Ctx_Cap := Configured_Ctx (M.Ctx);
      Ada.Text_IO.Put_Line
        ("  context window:" & Ctx_Cap'Image & " tokens (model"
         & M.Ctx'Image & ", set ASPIDA_CTX to change)");
      Ada.Text_IO.Flush;
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
                           R.Comp_Toks := Slots (S).N_Gen;   -- natural stop
                           R.Trunc     := False;
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
                           if Slots (S).N_Gen >= R.Max then
                              R.Comp_Toks := Slots (S).N_Gen;   -- output cap hit
                              R.Trunc     := True;
                              Retire (S);
                           elsif Slots (S).Pos >= Ctx_Cap then
                              --  Window full: roll it forward (context-shift)
                              --  and keep generating; only stop if no room can
                              --  be freed or shift is disabled.
                              if not Ctx_Shift_On
                                or else Shift_KV (S) = 0
                              then
                                 R.Comp_Toks := Slots (S).N_Gen;
                                 R.Trunc     := True;
                                 Retire (S);
                              end if;
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

   ---------------------------------------------------------------------
   -- KV_Prompt_Trim: pure prompt-clamp for the KV cache (Batch 1.4).
   ---------------------------------------------------------------------
   function KV_Prompt_Trim
     (Prompt  : LLM_Tokenizer.Token_Array;
      Ctx_Cap : Natural;
      Max     : Integer) return LLM_Tokenizer.Token_Array
   is
      use type LLM_Tokenizer.Token_Array;   --  make "&" on Token_Array visible
      Room           : constant Integer := Ctx_Cap - Integer'Max (1, Max) - 2;
      Cap_For_Prompt : constant Integer := Integer'Max (1, Ctx_Cap - 1);
   begin
      if Prompt'Length = 0 then
         return [];                            -- nothing to trim
      elsif Prompt'Length > Cap_For_Prompt then
         --  HARD: keep BOS + the most recent (Cap_For_Prompt-1) tokens.
         return Prompt (Prompt'First .. Prompt'First)
            & Prompt (Prompt'Last - (Cap_For_Prompt - 1) + 1 .. Prompt'Last);
      elsif Room > 1 and then Prompt'Length > Room then
         --  SOFT: keep BOS + the most recent (Room-1) tokens so Max still fits.
         return Prompt (Prompt'First .. Prompt'First)
            & Prompt (Prompt'Last - (Room - 1) + 1 .. Prompt'Last);
      else
         return Prompt;
      end if;
   end KV_Prompt_Trim;

   function Run_Request
     (M : Llama_Model; Prompt : LLM_Tokenizer.Token_Array;
      Max, Stop_A, Stop_B : Integer;
      Sink : access LLM_Qwen.Token_Sink'Class;
      Params : LLM_Sampler.Params;
      Stats : access LLM_Qwen.Gen_Stats := null) return String
   is
      procedure Free_R is new Ada.Unchecked_Deallocation (Request, Request_Acc);
      procedure Free_T is
        new Ada.Unchecked_Deallocation (LLM_Tokenizer.Token_Array, Sched_Tok_Ptr);
      Do_It : Boolean;
      R     : Request_Acc := new Request;
   begin
      Init_Guard.Claim (Do_It);
      if Do_It then Sched.Init (M); end if;
      --  Clamp so the prompt fits the slot's KV cache (see KV_Prompt_Trim for
      --  the two-case rationale: HARD clip to Ctx_Cap, SOFT trim to the
      --  generation window; BOS always pinned). The trim math is factored into
      --  a pure model-free function so test_kv_overflow can exercise the
      --  Prompt'Length > Ctx_Cap overflow path without a loaded model.
      declare
         Trimmed : constant LLM_Tokenizer.Token_Array :=
           KV_Prompt_Trim (Prompt, Ctx_Cap, Max);
      begin
         if Trimmed'Length < Prompt'Length then
            Ada.Text_IO.Put_Line
              ("  [ctx] prompt" & Prompt'Length'Image & " ->"
               & Trimmed'Length'Image & " (KV cap" & Ctx_Cap'Image
               & ", BOS pinned)");
         end if;
         R.Prompt := new LLM_Tokenizer.Token_Array'(Trimmed);
      end;
      R.Max := Max; R.Stop_A := Stop_A; R.Stop_B := Stop_B;
      R.Params := Params;
      R.Prompt_Toks := R.Prompt'Length;
      R.Sink_Addr := (if Sink /= null then Sink.all'Address
                      else System.Null_Address);
      Ada.Synchronous_Task_Control.Set_False (R.Done);
      Req_Queue.Put (R);
      Ada.Synchronous_Task_Control.Suspend_Until_True (R.Done);
      if Stats /= null then
         Stats.all := (Prompt_Tokens     => R.Prompt_Toks,
                       Completion_Tokens => R.Comp_Toks,
                       Truncated         => R.Trunc,
                       Overflow          => False);
      end if;
      return Res : constant String := To_String (R.Result) do
         Free_T (R.Prompt);
         Free_R (R);
      end return;
   end Run_Request;

   --------------------------------------------------------------------
   -- Greedy decode with incremental K/V cache.
   --------------------------------------------------------------------

   function Forward_Logits
     (M : Llama_Model; Ids : LLM_Tokenizer.Token_Array) return Logits_Flat
   is
      N     : constant Integer := Ids'Length;
      Vc    : constant Integer := M.Vocab;
      Cap   : constant Integer := Integer'Max (1, N);
      Cache : KV_Cache (1 .. M.N_Blocks);
      procedure Free is
        new Ada.Unchecked_Deallocation (Tensor_Array, Tensor_Array_Ptr);
   begin
      --  Extended return: build the [N*Vocab] result on the (heap-backed)
      --  secondary stack so a large vocab never lands a megabyte array on the
      --  primary stack.
      return Res : Logits_Flat (0 .. Integer'Max (0, N * Vc - 1)) do
         for L in 1 .. M.N_Blocks loop
            Cache (L).K := new Tensor_Array (1 .. Cap);
            Cache (L).V := new Tensor_Array (1 .. Cap);
         end loop;
         for P in 1 .. N loop
            LLM_Step_Lock.Acquire;
            begin
               declare
                  L : constant Tensor :=
                    Forward_Step (M, Cache, Ids (Ids'First + P - 1), P - 1);
               begin
                  for K in 1 .. Vc loop
                     Res ((P - 1) * Vc + (K - 1)) := Get_Flat (L, K);
                  end loop;
               end;
               LLM_Step_Lock.Release;
            exception
               when others => LLM_Step_Lock.Release; raise;
            end;
         end loop;
         for L in 1 .. M.N_Blocks loop
            Free (Cache (L).K);
            Free (Cache (L).V);
         end loop;
      end return;
   end Forward_Logits;

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
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access LLM_Qwen.Gen_Stats := null) return String
   is
      use type LLM_Tokenizer.Token_Array;
      use type LLM_Qwen.Role_Kind;
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

   begin
      --  Turn-aware context fitting: keep the system prompt (attention sink) +
      --  the most recent turns within the window, dropping the oldest. Falls
      --  back to a token-exact BOS-pinned trim inside Run_Request, and (in
      --  strict mode) refuses an over-window request so the server can return
      --  context_length_exceeded.
      declare
         N        : constant Natural := Conversation'Length;
         Sys_1st  : constant Boolean := N >= 1 and then
           Conversation (Conversation'First).Role = LLM_Qwen.Role_System;
         Overhead : constant Natural :=
           One (M.Bos)'Length + Header ("assistant")'Length;
         Budget   : constant Natural := Natural'Max
           (Overhead + 1,
            Integer'Max (1, Effective_Context (M)
                            - Integer'Max (1, Max_New_Tokens) - 4));
         Lengths  : Ctx_Window.Len_Array  (1 .. Integer'Max (1, N));
         Keep     : Ctx_Window.Keep_Array (1 .. Integer'Max (1, N)) :=
           [others => True];
         Ovf      : Boolean := False;

         --  Concatenate the kept messages, in order.
         function Kept_Ids (K : Positive) return LLM_Tokenizer.Token_Array is
         begin
            if K > N then return LLM_Tokenizer.Token_Array'(2 .. 1 => 0); end if;
            if Keep (K) then
               return Msg_Ids (Conversation (Conversation'First + K - 1))
                      & Kept_Ids (K + 1);
            else
               return Kept_Ids (K + 1);
            end if;
         end Kept_Ids;
      begin
         if N > 0 then
            for K in 1 .. N loop
               Lengths (K) :=
                 Msg_Ids (Conversation (Conversation'First + K - 1))'Length;
            end loop;
            Ctx_Window.Select_Messages
              (Lengths (1 .. N), Sys_1st, Overhead, Budget, Keep (1 .. N), Ovf);
         end if;

         if Ovf and then Strict_Ctx then
            if Stats /= null then Stats.Overflow := True; end if;
            return "";   -- server maps this to context_length_exceeded
         end if;

         return Run_Request
           (M, One (M.Bos) & Kept_Ids (1) & Header ("assistant"),
            Max_New_Tokens, -1, -1, Sink, Params, Stats);
      end;
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

   function Effective_Context (M : Llama_Model) return Integer is
     (Integer (Configured_Ctx (M.Ctx)));

end LLM_Llama;
