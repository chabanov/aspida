with Ada.Environment_Variables;
with Ada.Text_IO;
with Interfaces.C;
with LLM_Qwen_GPU;

package body LLM_Batcher is

   Max_Lanes : constant := 8;
   type Lane_Id is range 0 .. Max_Lanes - 1;

   --  Runtime config (set once by Configure).
   Cfg_NL    : Integer := 0;
   Cfg_Vocab : Integer := 0;
   Configured : Boolean := False with Volatile;

   Serve : constant Boolean :=
     Ada.Environment_Variables.Exists ("ASPIDA_BATCH_SERVE");

   function Enabled return Boolean is (Serve);

   type Int_Arr   is array (0 .. Max_Lanes - 1) of Integer;

   --  C-side scratch (heap, sized at Configure).
   type C_Int_Arr is array (Natural range <>) of Interfaces.C.int;
   type C_Int_Ptr is access C_Int_Arr;
   type F_Arr is array (Natural range <>) of Interfaces.C.C_float;
   type F_Ptr is access F_Arr;

   CRows, CPos : C_Int_Ptr;   -- [Max_Lanes]
   CHandles    : C_Int_Ptr;   -- [Max_Lanes * NL]
   Batch_Log   : F_Ptr;       -- [Max_Lanes * Vocab]

   Max_NL : constant := 64;
   type H_Copy_Arr is array (0 .. Max_NL - 1) of Interfaces.C.int;

   type Lane_Rec is record
      In_Use   : Boolean := False;
      Pending  : Boolean := False;
      Ready    : Boolean := False;
      Failed   : Boolean := False;  -- the forward covering this lane raised
      Row, Pos : Integer := 0;
      --  Handle VALUES copied at Post. The Driver used to read the handler
      --  task's stack array through an address — if the handler was aborted
      --  (client disconnect / step timeout) mid-batch, that read dangled and
      --  fed garbage handles to the GPU (wild pointers: the flaky illegal-
      --  access crashes and cross-lane state corruption). Values can't dangle.
      HV       : H_Copy_Arr := [others => 0];
      NL       : Integer := 0;
      --  Batch slot of the last forward covering this lane (-1 = none):
      --  Wait_Done copies the logits slice itself, inside the protected
      --  action, where the caller's buffer is guaranteed alive.
      BIdx     : Integer := -1;
      --  Generation epoch. Bumped on every Claim (and on Abandon). A batch's
      --  Take snapshots each lane's Seq; Mark_Done / the scatter only touch a
      --  lane whose Seq still matches — so a step whose handler already timed
      --  out and freed (or reused) the lane cannot wake the wrong generation
      --  or scatter into a buffer the abandoned handler has released.
      Seq      : Natural := 0;
   end record;
   type Lane_Array is array (Lane_Id) of Lane_Rec;

   --------------------------------------------------------------------------
   --  Pool — coordinates handler lanes with the single Driver task.
   --------------------------------------------------------------------------
   protected Pool is
      --  Blocks while every lane is in use: queued admission, never a fallback
      --  to the (unsafe-when-concurrent) single-request path.
      entry Claim (Lane : out Integer);
      procedure Free  (L : Integer);
      procedure Post  (L, Row, Pos, NL : Integer; H : System.Address);
      entry Wait_Done (Lane_Id) (Failed : out Boolean; Log : System.Address);
      entry Wait_Pending;                       -- gate: ≥1 lane posted
      function All_Pending return Boolean;       -- every active lane is pending
      --  Fills the package-level CHandles staging from the per-lane VALUE
      --  copies (never from caller memory).
      procedure Take (Lanes, Rows, Poss, NLs, Seqs : out Int_Arr;
                      N : out Integer);
      procedure Mark_Done (Lanes, Seqs : Int_Arr; N : Integer);
      --  The forward covering these lanes raised: wake the callers with the
      --  Failed flag so each aborts its own generation (Step raises).
      procedure Mark_Failed (Lanes, Seqs : Int_Arr; N : Integer);
      --  A handler timed out waiting for its step: drop any pending step and
      --  bump the epoch so the in-flight batch (if any) can no longer touch
      --  this lane. The handler then frees the lane and aborts.
      procedure Abandon (L : Integer);
   private
      S : Lane_Array;
      Pending_Count : Natural := 0;
      Active_Count  : Natural := 0;
   end Pool;

   protected body Pool is
      entry Claim (Lane : out Integer) when Active_Count < Max_Lanes is
      begin
         Lane := -1;
         for L in Lane_Id loop
            if not S (L).In_Use then
               --  New epoch for this generation (preserve the monotonic Seq
               --  across the reset so a stale in-flight step is rejected).
               S (L) := (In_Use => True, Seq => S (L).Seq + 1, others => <>);
               Lane := Integer (L);
               Active_Count := Active_Count + 1;
               exit;
            end if;
         end loop;
      end Claim;

      procedure Free (L : Integer) is
         Id : constant Lane_Id := Lane_Id (L);
         Keep : constant Natural := S (Id).Seq;
      begin
         if S (Id).Pending then Pending_Count := Pending_Count - 1; end if;
         if S (Id).In_Use then Active_Count := Active_Count - 1; end if;
         S (Id) := (In_Use => False, Seq => Keep, others => <>);
      end Free;

      procedure Abandon (L : Integer) is
         Id : constant Lane_Id := Lane_Id (L);
      begin
         if S (Id).Pending then
            S (Id).Pending := False;
            Pending_Count := Pending_Count - 1;
         end if;
         --  Invalidate any in-flight batch's claim on this lane.
         S (Id).Seq := S (Id).Seq + 1;
      end Abandon;

      procedure Post (L, Row, Pos, NL : Integer; H : System.Address) is
         Id : constant Lane_Id := Lane_Id (L);
         Src : C_Int_Arr (0 .. NL - 1) with Import, Address => H;
      begin
         S (Id).Row := Row; S (Id).Pos := Pos;
         S (Id).NL := Integer'Min (NL, Max_NL);
         for K in 0 .. S (Id).NL - 1 loop
            S (Id).HV (K) := Src (K);
         end loop;
         S (Id).BIdx := -1;
         S (Id).Ready := False; S (Id).Pending := True;
         Pending_Count := Pending_Count + 1;
      end Post;

      --  A handler blocks here until the forward covering its lane is done.
      entry Wait_Done (for L in Lane_Id) (Failed : out Boolean; Log : System.Address)
        when S (L).Ready is
      begin
         S (L).Ready  := False;
         Failed       := S (L).Failed;
         S (L).Failed := False;
         --  Copy this lane's logit slice HERE: the entry body is a protected
         --  action executed on the caller's behalf, so the caller cannot have
         --  unwound — its buffer is alive. (The Driver previously scattered
         --  into caller buffers from its own task: a use-after-free when a
         --  handler aborted between the Live check and the write.)
         if not Failed and then S (L).BIdx >= 0 then
            declare
               Dst : F_Arr (0 .. Cfg_Vocab - 1) with Import, Address => Log;
               Base : constant Natural := S (L).BIdx * Cfg_Vocab;
            begin
               Dst := Batch_Log (Base .. Base + Cfg_Vocab - 1);
            end;
         end if;
         S (L).BIdx := -1;
      end Wait_Done;

      --  The Driver waits here for the first pending lane, then (after a short
      --  coalescing delay taken outside the lock) drains all pending lanes into
      --  one batch. Without the delay the Driver grabs each request alone (B=1)
      --  because clients are out of phase — one streams its last token over the
      --  network while another is mid-forward.
      entry Wait_Pending when Pending_Count > 0 is
      begin
         null;
      end Wait_Pending;

      function All_Pending return Boolean is
      begin
         return Active_Count > 0 and then Pending_Count >= Active_Count;
      end All_Pending;

      procedure Take (Lanes, Rows, Poss, NLs, Seqs : out Int_Arr;
                      N : out Integer) is
      begin
         N := 0;
         for L in Lane_Id loop
            if S (L).Pending then
               Lanes (N) := Integer (L);
               Rows (N)  := S (L).Row;
               Poss (N)  := S (L).Pos;
               NLs (N)   := S (L).NL;
               Seqs (N)  := S (L).Seq;
               for K in 0 .. S (L).NL - 1 loop
                  CHandles (N * Cfg_NL + K) := S (L).HV (K);
               end loop;
               S (L).Pending := False;
               N := N + 1;
            end if;
         end loop;
         Pending_Count := 0;
      end Take;

      procedure Mark_Done (Lanes, Seqs : Int_Arr; N : Integer) is
      begin
         for I in 0 .. N - 1 loop
            --  Skip a lane whose handler timed out and freed/reused it (epoch
            --  changed) — waking it would hand the wrong generation these logits.
            if S (Lane_Id (Lanes (I))).Seq = Seqs (I) then
               S (Lane_Id (Lanes (I))).BIdx  := I;
               S (Lane_Id (Lanes (I))).Ready := True;
            end if;
         end loop;
      end Mark_Done;

      procedure Mark_Failed (Lanes, Seqs : Int_Arr; N : Integer) is
      begin
         for I in 0 .. N - 1 loop
            if S (Lane_Id (Lanes (I))).Seq = Seqs (I) then
               S (Lane_Id (Lanes (I))).Failed := True;
               S (Lane_Id (Lanes (I))).Ready  := True;
            end if;
         end loop;
      end Mark_Failed;
   end Pool;

   --------------------------------------------------------------------------
   --  Alloc lock (serialise per-generation GPU state allocation).
   --------------------------------------------------------------------------
   protected Alloc is
      entry Acquire;
      procedure Release;
   private
      Held : Boolean := False;
   end Alloc;

   protected body Alloc is
      entry Acquire when not Held is begin Held := True; end Acquire;
      procedure Release is begin Held := False; end Release;
   end Alloc;

   procedure Alloc_Lock is begin Alloc.Acquire; end Alloc_Lock;
   procedure Alloc_Unlock is begin Alloc.Release; end Alloc_Unlock;

   --------------------------------------------------------------------------
   --  Driver — the single GPU forward caller.
   --
   --  A task TYPE created on demand by Configure, not a library-level task
   --  object. As an object it was elaborated into every program that links the
   --  engine and then spun `exit when Configured; delay 0.02;` forever — 50
   --  wakeups/s in a server that never enables batching (ASPIDA_BATCH_SERVE
   --  unset, so Configure is never called), and, because a task blocked in a
   --  spin or on a protected entry can never reach a terminate alternative, it
   --  also kept the environment task from ever completing: every test binary
   --  that linked the engine hung at exit after passing, which is why make test
   --  never finished. Created here only once the batcher is really configured,
   --  a program that never batches never has it at all, and one that does gets
   --  exactly the loop below. The spin is gone with it: Configure allocates the
   --  buffers and publishes Configured BEFORE activating this task.
   --------------------------------------------------------------------------
   task type Driver_T;
   type Driver_Acc is access Driver_T;
   Drv : Driver_Acc;   -- null until the winning Configure creates it

   task body Driver_T is
      Lanes, Rows, Poss, NLs, Seqs : Int_Arr;
      N : Integer;
      Fwd_Count, Sum_B, Max_B : Integer := 0;
   begin
      loop
         Pool.Wait_Pending;
         --  Coalesce: poll up to ~6 ms for every active client to post, but
         --  commit early the instant they are all waiting. Trades a little
         --  first-token latency for a fuller batch (higher aggregate).
         declare
            Waited : Integer := 0;
         begin
            while not Pool.All_Pending and then Waited < 6 loop
               delay 0.001;
               Waited := Waited + 1;
            end loop;
         end;
         Pool.Take (Lanes, Rows, Poss, NLs, Seqs, N);
         if Ada.Environment_Variables.Exists ("ASPIDA_BATCH_LOG") then
            Fwd_Count := Fwd_Count + 1; Sum_B := Sum_B + N;
            if N > Max_B then Max_B := N; end if;
            if Fwd_Count mod 50 = 0 then
               Ada.Text_IO.Put_Line
                 (Ada.Text_IO.Standard_Error,
                  "[BATCHDRV] forwards=" & Integer'Image (Fwd_Count)
                  & " avg_B=" & Integer'Image (Sum_B / Fwd_Count)
                  & " max_B=" & Integer'Image (Max_B));
            end if;
         end if;
         --  The forward + scatter must NEVER kill this task: with the Driver
         --  dead every batch client would block in Wait_Done forever (a whole-
         --  server wedge). Chain_Forward_Batch can raise GPU_Error (the CUDA
         --  error check) — on any exception wake the covered lanes with the
         --  Failed flag so each caller aborts its own generation cleanly, and
         --  keep driving the next batch.
         begin
            --  Rows/positions staging (handles were copied by VALUE in Take —
            --  the Driver never dereferences a handler task's memory).
            for I in 0 .. N - 1 loop
               CRows (I) := Interfaces.C.int (Rows (I));
               CPos (I)  := Interfaces.C.int (Poss (I));
            end loop;
            LLM_Qwen_GPU.Chain_Forward_Batch
              (N, CRows.all'Address, CPos.all'Address, CHandles.all'Address,
               Batch_Log.all'Address);
            --  Delivery: Mark_Done records each live lane's batch slot; the
            --  logits are copied by Wait_Done itself, inside the protected
            --  action, where the caller's buffer is guaranteed alive.
            Pool.Mark_Done (Lanes, Seqs, N);
         exception
            when others =>
               Pool.Mark_Failed (Lanes, Seqs, N);
         end;
      end loop;
   end Driver_T;

   --------------------------------------------------------------------------
   --  Public API.
   --------------------------------------------------------------------------
   --  Configure runs on the HANDLER tasks (LLM_Qwen.Decode_Tokens calls it on
   --  the first batched generation), so `if Configured then return` alone never
   --  made it one-shot: two handlers could both read False and both allocate.
   --  That was survivable while it only leaked a duplicate buffer set; it is
   --  not now that the winner also activates the Driver, since a second Driver
   --  would race the first for the same lanes. Setup_Gate hands exactly one
   --  caller the job. Claim only flips a flag — the allocation and the task
   --  activation stay OUTSIDE the protected action (activating a task inside
   --  one is a bounded error, and `new` under a lock would serialise handlers
   --  for no reason).
   protected Setup_Gate is
      procedure Claim (Won : out Boolean);
   private
      Taken : Boolean := False;
   end Setup_Gate;

   protected body Setup_Gate is
      procedure Claim (Won : out Boolean) is
      begin
         Won   := not Taken;
         Taken := True;
      end Claim;
   end Setup_Gate;

   procedure Configure (N_Layers, Vocab : Integer) is
      Won : Boolean;
   begin
      if Configured then return; end if;
      Setup_Gate.Claim (Won);
      if not Won then
         --  Someone else is mid-setup. Wait for it to publish rather than
         --  returning early: the caller goes straight on to Begin_Gen and the
         --  Driver it wakes reads Batch_Log / CHandles.
         while not Configured loop
            delay 0.001;
         end loop;
         return;
      end if;
      Cfg_NL := N_Layers; Cfg_Vocab := Vocab;
      CRows    := new C_Int_Arr (0 .. Max_Lanes - 1);
      CPos     := new C_Int_Arr (0 .. Max_Lanes - 1);
      CHandles := new C_Int_Arr (0 .. Max_Lanes * N_Layers - 1);
      Batch_Log := new F_Arr (0 .. Max_Lanes * Vocab - 1);
      --  Publish the buffers BEFORE activating the Driver: it dereferences
      --  them as soon as it has a batch, and it no longer waits on Configured.
      Configured := True;
      --  Setup_Gate already makes this one-shot; the null test keeps "exactly
      --  one Driver" checkable here rather than only in the gate's history.
      if Drv = null then
         Drv := new Driver_T;
      end if;
   end Configure;

   procedure Begin_Gen (Lane : out Integer) is
   begin
      Pool.Claim (Lane);
   end Begin_Gen;

   procedure End_Gen (Lane : Integer) is
   begin
      if Lane >= 0 then Pool.Free (Lane); end if;
   end End_Gen;

   --  Per-token liveness deadline. A single batched forward is milliseconds;
   --  if a lane's step is not served within this many seconds it is wedged (a
   --  lost wakeup / stuck forward), so the handler abandons it, FREES ITS LANE
   --  and aborts — turning what used to be a permanent whole-server wedge
   --  (leaked lanes until all 8 are gone) into one failed request. 0 disables
   --  the watchdog (legacy unbounded wait). Env ASPIDA_GEN_STEP_TIMEOUT_S.
   function Read_Step_Timeout return Duration is
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_GEN_STEP_TIMEOUT_S") then
         return Duration'Value
           (Ada.Environment_Variables.Value ("ASPIDA_GEN_STEP_TIMEOUT_S"));
      end if;
      return 30.0;
   exception
      when others => return 30.0;
   end Read_Step_Timeout;

   Step_Timeout : constant Duration := Read_Step_Timeout;

   procedure Step (Lane, Embed_Row, Pos, N_Layers : Integer;
                   Handles, Logits : System.Address) is
      Failed : Boolean;
   begin
      Pool.Post (Lane, Embed_Row, Pos, N_Layers, Handles);
      if Step_Timeout > 0.0 then
         select
            Pool.Wait_Done (Lane_Id (Lane)) (Failed, Logits);
         or
            delay Step_Timeout;
            --  Stuck: drop this lane's pending step, bump its epoch so the
            --  in-flight batch (if any) can no longer scatter into our buffer
            --  or wake us, then abort. End_Gen frees the lane on unwind.
            Pool.Abandon (Lane);
            raise Batch_Failed with "batched forward timed out (lane wedged)";
         end select;
      else
         Pool.Wait_Done (Lane_Id (Lane)) (Failed, Logits);
      end if;
      if Failed then
         raise Batch_Failed with "batched GPU forward failed";
      end if;
   end Step;

end LLM_Batcher;
