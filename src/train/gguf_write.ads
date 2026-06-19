---------------------------------------------------------------------
-- GGUF_Write — serialize metadata + F32 tensors into a GGUF v3 file that the
-- Aspida inference engine (LLM_GGUF) can read back.
--
-- This closes the teach->train->serve loop: a student trained by Train can be
-- written here and then loaded by the same engine that produced its teacher.
-- Layout mirrors LLM_GGUF.Open byte-for-byte (little-endian):
--   "GGUF", u32 version=3, u64 tensor_count, u64 meta_count,
--   meta KVs (key:str, type:u32, value), tensor infos (name, n_dims, dims,
--   type:u32, offset:u64), pad to alignment (32), tensor data.
---------------------------------------------------------------------

with Ada.Strings.Unbounded;
with Ada.Containers.Vectors;

package GGUF_Write is

   type Builder is limited private;

   type Dims_Array  is array (Positive range <>) of Natural;
   type Float_Array is array (Positive range <>) of Float;
   type Str_List    is array (Positive range <>) of Ada.Strings.Unbounded.Unbounded_String;

   --  Metadata entries (added in order; written in that order).
   procedure Meta_Str       (B : in out Builder; Key, Val : String);
   procedure Meta_U32       (B : in out Builder; Key : String; Val : Natural);
   procedure Meta_F32       (B : in out Builder; Key : String; Val : Float);
   procedure Meta_Str_Array (B : in out Builder; Key : String; Vals : Str_List);

   --  Append an F32 tensor. Data is a flat row-major buffer; Dims are the GGUF
   --  ne[] (ne0 fastest-varying), product must equal Data'Length.
   procedure Add_Tensor_F32
     (B : in out Builder; Name : String; Dims : Dims_Array; Data : Float_Array)
     with Pre => Dims'Length >= 1;

   --  Append an already-quantized tensor (raw ggml bytes, e.g. from
   --  LLM_Quant.Quantize_*). Dims are the logical ne[] (element counts);
   --  ne0 must be a multiple of the block size (32).
   procedure Add_Tensor_Q8_0
     (B : in out Builder; Name : String; Dims : Dims_Array; Raw : String)
     with Pre => Dims'Length >= 1;
   procedure Add_Tensor_Q4_0
     (B : in out Builder; Name : String; Dims : Dims_Array; Raw : String)
     with Pre => Dims'Length >= 1;
   procedure Add_Tensor_Q5_0
     (B : in out Builder; Name : String; Dims : Dims_Array; Raw : String)
     with Pre => Dims'Length >= 1;
   procedure Add_Tensor_Q4_K
     (B : in out Builder; Name : String; Dims : Dims_Array; Raw : String)
     with Pre => Dims'Length >= 1;
   procedure Add_Tensor_Q5_K
     (B : in out Builder; Name : String; Dims : Dims_Array; Raw : String)
     with Pre => Dims'Length >= 1;
   procedure Add_Tensor_Q6_K
     (B : in out Builder; Name : String; Dims : Dims_Array; Raw : String)
     with Pre => Dims'Length >= 1;

   --  Write the assembled GGUF file.
   procedure Save (B : in out Builder; Path : String);

private

   use Ada.Strings.Unbounded;

   type Dim4 is array (1 .. 4) of Natural;

   type Tensor_Rec is record
      Name   : Unbounded_String;
      N_Dims : Natural := 0;
      Dims   : Dim4 := [others => 0];
      Kind   : Natural := 0;        -- GGML type code (0 = F32, 8 = Q8_0)
      Data   : Unbounded_String;    -- raw little-endian tensor bytes
   end record;

   --  growable byte buffers held as Unbounded_String (1 char == 1 byte)
   type Meta_Buf is record
      Bytes : Unbounded_String;
      Count : Natural := 0;
   end record;

   package Tensor_Vectors is
     new Ada.Containers.Vectors (Positive, Tensor_Rec);

   type Builder is record
      Meta : Meta_Buf;
      Tens : Tensor_Vectors.Vector;
   end record;

end GGUF_Write;
