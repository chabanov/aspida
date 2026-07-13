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
   type Addr_Arr  is array (0 .. Max_Lanes - 1) of System.Address;

   --  C-side scratch (heap, sized at Configure).
   type C_Int_Arr is array (Natural range <>) of Interfaces.C.int;
   type C_Int_Ptr is access C_Int_Arr;
   type F_Arr is array (Natural range <>) of Interfaces.C.C_float;
   type F_Ptr is access F_Arr;

   CRows, CPos : C_Int_Ptr;   -- [Max_Lanes]
   CHandles    : C_Int_Ptr;   -- [Max_Lanes * NL]
   Batch_Log   : F_Ptr;       -- [Max_Lanes * Vocab]

   type Lane_Rec is record
      In_Use   : Boolean := False;
      Pending  : Boolean := False;
      Ready    : Boolean := False;
      Row, Pos : Integer := 0;
      H_Addr   : System.Address := System.Null_Address;
      NL       : Integer := 0;
      Log_Addr : System.Address := System.Null_Address;
   end record;
   type Lane_Array is array (Lane_Id) of Lane_Rec;

   --------------------------------------------------------------------------
   --  Pool — coordinates handler lanes with the single Driver task.
   --------------------------------------------------------------------------
   protected Pool is
      procedure Claim (Lane : out Integer);
      procedure Free  (L : Integer);
      procedure Post  (L, Row, Pos, NL : Integer; H, Lg : System.Address);
      entry Wait_Done (Lane_Id);
      entry Wait_Pending;                       -- gate: ≥1 lane posted
      function All_Pending return Boolean;       -- every active lane is pending
      procedure Take (Lanes, Rows, Poss, NLs : out Int_Arr;
                      H, Logs : out Addr_Arr; N : out Integer);
      procedure Mark_Done (Lanes : Int_Arr; N : Integer);
   private
      S : Lane_Array;
      Pending_Count : Natural := 0;
      Active_Count  : Natural := 0;
   end Pool;

   protected body Pool is
      procedure Claim (Lane : out Integer) is
      begin
         Lane := -1;
         for L in Lane_Id loop
            if not S (L).In_Use then
               S (L) := (In_Use => True, others => <>);
               Lane := Integer (L);
               Active_Count := Active_Count + 1;
               exit;
            end if;
         end loop;
      end Claim;

      procedure Free (L : Integer) is
      begin
         if S (Lane_Id (L)).Pending then Pending_Count := Pending_Count - 1; end if;
         if S (Lane_Id (L)).In_Use then Active_Count := Active_Count - 1; end if;
         S (Lane_Id (L)) := (In_Use => False, others => <>);
      end Free;

      procedure Post (L, Row, Pos, NL : Integer; H, Lg : System.Address) is
         Id : constant Lane_Id := Lane_Id (L);
      begin
         S (Id).Row := Row; S (Id).Pos := Pos; S (Id).NL := NL;
         S (Id).H_Addr := H; S (Id).Log_Addr := Lg;
         S (Id).Ready := False; S (Id).Pending := True;
         Pending_Count := Pending_Count + 1;
      end Post;

      --  A handler blocks here until the forward covering its lane is done.
      entry Wait_Done (for L in Lane_Id) when S (L).Ready is
      begin
         S (L).Ready := False;
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

      procedure Take (Lanes, Rows, Poss, NLs : out Int_Arr;
                      H, Logs : out Addr_Arr; N : out Integer) is
      begin
         N := 0;
         for L in Lane_Id loop
            if S (L).Pending then
               Lanes (N) := Integer (L);
               Rows (N)  := S (L).Row;
               Poss (N)  := S (L).Pos;
               NLs (N)   := S (L).NL;
               H (N)     := S (L).H_Addr;
               Logs (N)  := S (L).Log_Addr;
               S (L).Pending := False;
               N := N + 1;
            end if;
         end loop;
         Pending_Count := 0;
      end Take;

      procedure Mark_Done (Lanes : Int_Arr; N : Integer) is
      begin
         for I in 0 .. N - 1 loop
            S (Lane_Id (Lanes (I))).Ready := True;
         end loop;
      end Mark_Done;
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
   --------------------------------------------------------------------------
   task Driver;

   task body Driver is
      Lanes, Rows, Poss, NLs : Int_Arr;
      H, Logs : Addr_Arr;
      N : Integer;
      Fwd_Count, Sum_B, Max_B : Integer := 0;
   begin
      loop
         exit when Configured;
         delay 0.02;
      end loop;
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
         Pool.Take (Lanes, Rows, Poss, NLs, H, Logs, N);
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
         --  Assemble the batched inputs from each lane's snapshot.
         for I in 0 .. N - 1 loop
            CRows (I) := Interfaces.C.int (Rows (I));
            CPos (I)  := Interfaces.C.int (Poss (I));
            declare
               Src : C_Int_Arr (0 .. NLs (I) - 1) with Import, Address => H (I);
            begin
               CHandles (I * Cfg_NL .. I * Cfg_NL + NLs (I) - 1) := Src;
            end;
         end loop;
         LLM_Qwen_GPU.Chain_Forward_Batch
           (N, CRows.all'Address, CPos.all'Address, CHandles.all'Address,
            Batch_Log.all'Address);
         --  Scatter each lane's logit slice back to its caller's buffer.
         for I in 0 .. N - 1 loop
            declare
               Dst : F_Arr (0 .. Cfg_Vocab - 1) with Import, Address => Logs (I);
            begin
               Dst := Batch_Log (I * Cfg_Vocab .. I * Cfg_Vocab + Cfg_Vocab - 1);
            end;
         end loop;
         Pool.Mark_Done (Lanes, N);
      end loop;
   end Driver;

   --------------------------------------------------------------------------
   --  Public API.
   --------------------------------------------------------------------------
   procedure Configure (N_Layers, Vocab : Integer) is
   begin
      if Configured then return; end if;
      Cfg_NL := N_Layers; Cfg_Vocab := Vocab;
      CRows    := new C_Int_Arr (0 .. Max_Lanes - 1);
      CPos     := new C_Int_Arr (0 .. Max_Lanes - 1);
      CHandles := new C_Int_Arr (0 .. Max_Lanes * N_Layers - 1);
      Batch_Log := new F_Arr (0 .. Max_Lanes * Vocab - 1);
      Configured := True;
   end Configure;

   procedure Begin_Gen (Lane : out Integer) is
   begin
      Pool.Claim (Lane);
   end Begin_Gen;

   procedure End_Gen (Lane : Integer) is
   begin
      if Lane >= 0 then Pool.Free (Lane); end if;
   end End_Gen;

   procedure Step (Lane, Embed_Row, Pos, N_Layers : Integer;
                   Handles, Logits : System.Address) is
   begin
      Pool.Post (Lane, Embed_Row, Pos, N_Layers, Handles, Logits);
      Pool.Wait_Done (Lane_Id (Lane));
   end Step;

end LLM_Batcher;
