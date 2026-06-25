------------------------------------------------------------------------
-- Train_GPU_Resident — Ada binding to the RESIDENT training-session shim
-- (libaspidatrain.so, see gpu/train_resident_shim.cu), dlopen'd at runtime
-- exactly like LLM_GPU. Off unless the shim is present; path from
-- ASPIDA_TRAIN_LIB (default ./libaspidatrain.so). A session keeps weights,
-- AdamW moments and data resident on the device across Steps; only the scalar
-- loss crosses the FFI per step (no per-op host round-trips).
--
-- This is Step 5b: Ada drives GPU-resident training. Step 5c grad-checks it,
-- then the full Student forward/backward is grafted on.
------------------------------------------------------------------------

with System;
with Interfaces.C;

package Train_GPU_Resident is

   subtype C_Float is Interfaces.C.C_float;
   type F32_Array is array (Natural range <>) of C_Float;

   --  Opaque device-resident session handle.
   subtype Session is System.Address;

   --  True once the shim is dlopen'd and all entry points resolved.
   function Available return Boolean;

   Not_Available : exception;   -- raised by the ops when Available is False

   --  Create a resident L-layer (DxD) linear-stack session, batch B.
   function Create (L, B, D : Integer; LR : Float) return Session
     with Pre => L >= 1 and then B >= 1 and then D >= 1;

   --  Upload input X[B*D] and target T[B*D] ONCE (they stay resident).
   procedure Set_Data (S : Session; X, T : F32_Array);

   --  One resident forward+backward+AdamW step; returns loss/element.
   function Step (S : Session) return Float;

   --  Release the session's device memory.
   procedure Free (S : Session);

   --  Grad-check support (Step 5c): forward-only loss E = ½·Σ(Y−T)², the
   --  analytic gradient dE/dW[Layer][Idx] (no AdamW update), and single-weight
   --  read/perturb — so the GPU backward can be checked against a CPU-side
   --  finite difference of the same loss.
   function  Loss_Only (S : Session) return Float;
   function  Grad_At  (S : Session; Layer, Idx : Integer) return Float;
   function  W_Get    (S : Session; Layer, Idx : Integer) return Float;
   procedure W_Set    (S : Session; Layer, Idx : Integer; V : Float);

end Train_GPU_Resident;
