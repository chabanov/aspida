------------------------------------------------------------------------
-- Student_GPU — Ada binding to the resident full-transformer Student session
-- (libaspidastudent.so, see gpu/student_shim.cu), dlopen'd at runtime like
-- LLM_GPU / Train_GPU_Resident. A session keeps the whole Student (embedding,
-- transformer layers, head) + grads + activations resident on the device; each
-- Step is a full forward + backward + SGD update on the GPU and returns only the
-- scalar loss. Off (CPU fallback) unless the shim is present; path from
-- ASPIDA_STUDENT_LIB (default ./libaspidastudent.so).
--
-- Step 5i Stage C2: Ada drives full transformer-LM training on the GPU.
------------------------------------------------------------------------

with System;
with Interfaces.C;

package Student_GPU is

   type Int_Array is array (Natural range <>) of Interfaces.C.int;

   subtype Session is System.Address;

   function Available return Boolean;
   Not_Available : exception;

   --  Create a resident Student of the given architecture (maps to a
   --  Platform.Student_Tier via Config_Of). Returns Null_Address if the shim
   --  rejects the config (e.g. L too large, D not divisible by H).
   function Create (Voc, Dim, Ff, Seq, Layers, Heads : Positive) return Session
     with Pre => Dim mod Heads = 0;

   --  Upload one next-token example: token Ids and Targets (length = the shim's
   --  sequence length S). They stay resident.
   procedure Set_Data (S : Session; Ids, Targets : Int_Array)
     with Pre => Ids'Length = Targets'Length;

   type F32_Array is array (Natural range <>) of Interfaces.C.C_float;

   --  Upload one DISTILLATION example: token Ids and a per-position teacher
   --  distribution Q[S*V] (each of the S rows sums to 1); switches the loss to
   --  soft-target (KL) cross-entropy against the teacher. This is how the
   --  teachers' knowledge reaches the GPU student.
   procedure Set_Distill (S : Session; Ids : Int_Array; Q : F32_Array);

   --  One resident forward + backward + AdamW step; returns the cross-entropy loss.
   function Step (S : Session; LR : Float) return Float;

   --  Total trainable-parameter count (sum over every weight tensor) and a
   --  flat read-back of the trained weights in opt-register order (per layer:
   --  Wq,Wk,Wv,Wo,G1,G2,Wg,Wu,Wd; then Gf,Wh,E). The bridge that lets the host
   --  export a servable GGUF of the GPU-resident model. Out'Length must equal
   --  N_Params (S).
   function  N_Params   (S : Session) return Natural;
   procedure Get_Weights (S : Session; Out_W : out F32_Array)
     with Pre => Out_W'Length = N_Params (S);

   --  Gradient accumulation: Micro accumulates one micro-batch's gradients (call
   --  Set_Data between micro-batches); Apply averages over G and does one AdamW
   --  update. Amortises an all-reduce over G micro-steps for data-parallel.
   function  Micro (S : Session) return Float;
   procedure Apply (S : Session; LR : Float; G : Positive);

   procedure Free (S : Session);

end Student_GPU;
