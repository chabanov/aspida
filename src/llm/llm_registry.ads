---------------------------------------------------------------------
--  LLM_Registry — multi-model serving (one server, N models).
--
--  A thread-safe, lazy-loading, bounded, refcounted map from a model ref
--  (the OpenAI request's `model` field — a catalog id or path) to a loaded
--  LLM_Engine.Engine, so one secure_server serves many models routed by ref
--  over the same E2EE channel (see docs/MULTI_MODEL_SERVING.md).
--
--  Bounded with LRU eviction (Phase 1b). At most ASPIDA_MAX_LOADED_MODELS
--  resident (default 3); the default model is always warm (slot 1, never
--  evicted). When a cold model is requested and every slot is full, the
--  least-recently-used slot whose refcount = 0 (and which is not the default)
--  is unloaded — its backend Released (weights + GPU mirror + file handles
--  freed via LLM_Engine.Unload, no leak) — and reused for the new model. If
--  EVERY non-default slot is pinned (refcount > 0, i.e. in flight), eviction is
--  impossible and the request fails loud (Ok=False) — never a silent
--  wrong-model answer, never evicting an in-use model.
--
--  Concurrency: lookups of already-loaded models are lock-light; model LOADS
--  are serialized (loading two large models at once would OOM) — concurrent
--  requests for a cold ref wait behind the single in-progress load.
---------------------------------------------------------------------

with LLM_Engine;
with Ada.Strings.Unbounded;

package LLM_Registry is

   --  A refcounted lease on a loaded model. While held, the engine is pinned.
   --  Release each acquired lease exactly once (Engine_Of is valid until then).
   type Lease is private;

   --  Seed the registry with the always-warm default model in slot 1.
   --  Default_Ref is the id under which it is addressed (its catalog basename);
   --  an empty request `model` also resolves here. Idempotent re-init replaces.
   procedure Init (Default_Ref : String; Default : LLM_Engine.Engine);

   --  Lease the engine serving Ref (catalog id or full path; empty => default).
   --  Loads on demand (serialized) up to the budget. On success Ok=True and the
   --  lease pins the engine. On failure Ok=False and Err explains:
   --    * the ref cannot be resolved to a supported GGUF, or
   --    * the server is at model capacity and Ref is not already warm.
   procedure Acquire
     (Ref : String;
      L   : out Lease;
      Ok  : out Boolean;
      Err : out Ada.Strings.Unbounded.Unbounded_String);

   --  The engine behind a held lease. Precondition: a successful Acquire.
   function Engine_Of (L : Lease) return LLM_Engine.Engine;

   --  Drop a lease (decrement the model's refcount). Safe on a default lease.
   procedure Release (L : in out Lease);

   --  Observability.
   function Loaded_Count return Natural;   --  models currently resident
   function Max_Models   return Natural;   --  ASPIDA_MAX_LOADED_MODELS

private

   --  A lease is just the slot index it pins (0 = invalid / not held).
   type Lease is record
      Slot : Natural := 0;
   end record;

end LLM_Registry;
