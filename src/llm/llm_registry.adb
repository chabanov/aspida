---------------------------------------------------------------------
--  LLM_Registry body — see the spec + docs/MULTI_MODEL_SERVING.md.
--
--  Two protected objects:
--    * Map       — the slot table; fast, non-blocking lookups/refcount.
--    * Load_Gate — a 1-slot gate that serializes model LOADS (concurrent
--                  loads of large models would OOM); the slow LLM_Engine.Load
--                  runs OUTSIDE Map while holding the gate.
--  v1 has no eviction, so a committed engine lives for the server's lifetime —
--  the Engine value copied out by Engine_Of can therefore be used outside the
--  lock without any dangling-access risk.
---------------------------------------------------------------------

with Ada.Environment_Variables;
with Ada.Directories;
with Ada.Strings.Unbounded;  use Ada.Strings.Unbounded;
with LLM_Catalog;

package body LLM_Registry is

   --  ASPIDA_MAX_LOADED_MODELS (default 3, clamped to [1, 64]).
   function Env_Max return Positive is
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_MAX_LOADED_MODELS") then
         declare
            N : constant Integer :=
              Integer'Value (Ada.Environment_Variables.Value
                               ("ASPIDA_MAX_LOADED_MODELS"));
         begin
            if N < 1 then return 1;
            elsif N > 64 then return 64;
            else return N;
            end if;
         end;
      end if;
      return 3;
   exception
      when others => return 3;
   end Env_Max;

   Cap         : constant Positive := Env_Max;
   Default_Key : Unbounded_String := Null_Unbounded_String;

   type Slot_Rec is record
      Ref      : Unbounded_String := Null_Unbounded_String;
      Eng      : LLM_Engine.Engine;
      Loaded   : Boolean := False;
      Refs     : Natural := 0;
      Last_Use : Long_Long_Integer := 0;  --  monotonic tick of last Acquire/Commit
   end record;

   type Slot_Array is array (Positive range <>) of Slot_Rec;

   ------------------------------------------------------------------
   --  Slot table.
   ------------------------------------------------------------------
   protected Map is
      procedure Seed_Default (Key : String; E : LLM_Engine.Engine);
      --  Ref + pin a slot whose Ref = Key, if loaded. Slot=0 when not present.
      procedure Find_And_Ref (Key : String; Slot : out Natural);
      --  First not-loaded slot, or 0 when at capacity.
      procedure Find_Free (Slot : out Natural);
      --  Pick + claim the least-recently-used evictable slot: Loaded, Refs = 0,
      --  not the default slot (1). Marks it not-loaded (so no concurrent Acquire
      --  can match it) and returns its Engine in Victim_Eng for the caller to
      --  Unload OUTSIDE the lock. Slot = 0 when nothing is evictable (every
      --  non-default slot is pinned) — the caller then fails loud.
      procedure Claim_LRU_Victim
        (Slot : out Natural; Victim_Eng : out LLM_Engine.Engine);
      procedure Commit (Slot : Natural; Key : String; E : LLM_Engine.Engine);
      procedure Unref (Slot : Natural);
      function  Get_Engine (Slot : Natural) return LLM_Engine.Engine;
      function  Count return Natural;
   private
      Slots : Slot_Array (1 .. Cap);
      Clock : Long_Long_Integer := 0;   --  monotonic last-use tick source
   end Map;

   protected body Map is

      procedure Seed_Default (Key : String; E : LLM_Engine.Engine) is
      begin
         Clock := Clock + 1;
         Slots (1) := (Ref      => To_Unbounded_String (Key),
                       Eng      => E,
                       Loaded   => True,
                       Refs     => 0,
                       Last_Use => Clock);
      end Seed_Default;

      procedure Find_And_Ref (Key : String; Slot : out Natural) is
      begin
         Slot := 0;
         for I in Slots'Range loop
            if Slots (I).Loaded and then To_String (Slots (I).Ref) = Key then
               Slots (I).Refs := Slots (I).Refs + 1;
               Clock := Clock + 1;
               Slots (I).Last_Use := Clock;   --  most-recently used
               Slot := I;
               return;
            end if;
         end loop;
      end Find_And_Ref;

      procedure Find_Free (Slot : out Natural) is
      begin
         Slot := 0;
         for I in Slots'Range loop
            if not Slots (I).Loaded then
               Slot := I;
               return;
            end if;
         end loop;
      end Find_Free;

      procedure Claim_LRU_Victim
        (Slot : out Natural; Victim_Eng : out LLM_Engine.Engine)
      is
         Best : Natural := 0;
      begin
         Slot := 0;
         --  Never the default slot (1); only fully-released (Refs = 0) loaded
         --  slots are evictable. Among those, the smallest Last_Use is the LRU.
         for I in 2 .. Slots'Last loop
            if Slots (I).Loaded and then Slots (I).Refs = 0 then
               if Best = 0
                 or else Slots (I).Last_Use < Slots (Best).Last_Use
               then
                  Best := I;
               end if;
            end if;
         end loop;

         if Best = 0 then
            return;   --  nothing evictable: every non-default slot is pinned
         end if;

         --  Claim it: hand the Engine out for an out-of-lock Unload and mark the
         --  slot free so no concurrent Acquire can match (or re-pin) it. The
         --  stored Eng is left as-is until Commit overwrites the whole record;
         --  the copy returned in Victim_Eng owns the teardown.
         Victim_Eng := Slots (Best).Eng;
         Slots (Best).Loaded := False;
         Slots (Best).Ref    := Null_Unbounded_String;
         Slot := Best;
      end Claim_LRU_Victim;

      procedure Commit (Slot : Natural; Key : String; E : LLM_Engine.Engine) is
      begin
         Clock := Clock + 1;
         Slots (Slot) := (Ref      => To_Unbounded_String (Key),
                          Eng      => E,
                          Loaded   => True,
                          Refs     => 1,
                          Last_Use => Clock);
      end Commit;

      procedure Unref (Slot : Natural) is
      begin
         if Slot in Slots'Range and then Slots (Slot).Refs > 0 then
            Slots (Slot).Refs := Slots (Slot).Refs - 1;
         end if;
      end Unref;

      function Get_Engine (Slot : Natural) return LLM_Engine.Engine is
      begin
         return Slots (Slot).Eng;
      end Get_Engine;

      function Count return Natural is
         N : Natural := 0;
      begin
         for I in Slots'Range loop
            if Slots (I).Loaded then N := N + 1; end if;
         end loop;
         return N;
      end Count;

   end Map;

   ------------------------------------------------------------------
   --  Load serialization (one in-flight model load at a time).
   ------------------------------------------------------------------
   protected Load_Gate is
      entry Enter;
      procedure Leave;
   private
      Busy : Boolean := False;
   end Load_Gate;

   protected body Load_Gate is
      entry Enter when not Busy is
      begin
         Busy := True;
      end Enter;
      procedure Leave is
      begin
         Busy := False;
      end Leave;
   end Load_Gate;

   ------------------------------------------------------------------
   --  Resolve a ref (catalog id / basename / full path) to a supported
   --  GGUF path. Mirrors secure_server's Tag_Select resolution.
   ------------------------------------------------------------------
   procedure Resolve
     (Ref : String; Path : out Unbounded_String; Found : out Boolean)
   is
      use type LLM_Catalog.Model_Status;
      Cat : constant LLM_Catalog.Entry_Vectors.Vector := LLM_Catalog.Discover;
   begin
      Found := False;
      Path  := Null_Unbounded_String;
      for E of Cat loop
         if (To_String (E.Path) = Ref
             or else Ada.Directories.Simple_Name (To_String (E.Path)) = Ref)
           and then E.Status = LLM_Catalog.Supported
         then
            Found := True;
            Path  := E.Path;
            return;
         end if;
      end loop;
   end Resolve;

   ------------------------------------------------------------------
   --  Public API.
   ------------------------------------------------------------------

   procedure Init (Default_Ref : String; Default : LLM_Engine.Engine) is
   begin
      Default_Key := To_Unbounded_String (Default_Ref);
      Map.Seed_Default (Default_Ref, Default);
   end Init;

   procedure Acquire
     (Ref : String;
      L   : out Lease;
      Ok  : out Boolean;
      Err : out Ada.Strings.Unbounded.Unbounded_String)
   is
      Key  : constant String :=
        (if Ref = "" then To_String (Default_Key) else Ref);
      Slot : Natural;
   begin
      Err := Null_Unbounded_String;
      L   := (Slot => 0);

      --  Fast path: already resident.
      Map.Find_And_Ref (Key, Slot);
      if Slot /= 0 then
         L  := (Slot => Slot);
         Ok := True;
         return;
      end if;

      --  Cold: serialize the load.
      Load_Gate.Enter;

      --  Re-check — another task may have loaded Key while we waited.
      Map.Find_And_Ref (Key, Slot);
      if Slot /= 0 then
         Load_Gate.Leave;
         L  := (Slot => Slot);
         Ok := True;
         return;
      end if;

      --  Resolve the ref BEFORE touching capacity, so an unknown model never
      --  evicts a resident one (fail loud without disturbing the warm set).
      declare
         Path    : Unbounded_String;
         Resolved : Boolean;
      begin
         Resolve (Key, Path, Resolved);
         if not Resolved then
            Load_Gate.Leave;
            Ok  := False;
            Err := To_Unbounded_String ("unknown or unsupported model: " & Key);
            return;
         end if;

         --  Capacity: take a free slot, else evict the LRU unpinned slot.
         Map.Find_Free (Slot);
         if Slot = 0 then
            declare
               Victim : LLM_Engine.Engine;
            begin
               Map.Claim_LRU_Victim (Slot, Victim);
               if Slot = 0 then
                  Load_Gate.Leave;
                  Ok  := False;
                  Err := To_Unbounded_String
                    ("server at model capacity (" & Natural'Image (Map.Count)
                     & " loaded, all in use); model """ & Key
                     & """ is not warm and no idle model can be evicted");
                  return;
               end if;
               --  Tear down the evicted model OUTSIDE the lock (frees its
               --  weights, GPU mirror and file handles; can be slow). The slot
               --  is already marked free, so the load below reuses it.
               LLM_Engine.Unload (Victim);
            end;
         end if;

         declare
            E : constant LLM_Engine.Engine :=
              LLM_Engine.Load (To_String (Path));
         begin
            Map.Commit (Slot, Key, E);
            Load_Gate.Leave;
            L  := (Slot => Slot);
            Ok := True;
         end;
      exception
         when others =>
            Load_Gate.Leave;
            Ok  := False;
            Err := To_Unbounded_String ("failed to load model: " & Key);
      end;
   end Acquire;

   function Engine_Of (L : Lease) return LLM_Engine.Engine is
   begin
      return Map.Get_Engine (L.Slot);
   end Engine_Of;

   procedure Release (L : in out Lease) is
   begin
      if L.Slot /= 0 then
         Map.Unref (L.Slot);
         L.Slot := 0;
      end if;
   end Release;

   function Loaded_Count return Natural is (Map.Count);
   function Max_Models   return Natural is (Cap);

end LLM_Registry;
