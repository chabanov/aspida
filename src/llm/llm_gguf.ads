---------------------------------------------------------------------
-- LLM_GGUF — GGUF v3 file parser
--
-- GGUF is the binary format used by llama.cpp for storing quantized
-- transformer models. This package reads metadata, tensor descriptors,
-- and provides raw byte access to tensor data for dequantization.
--
-- GGUF v3 layout:
--   [0..3]   magic: "GGUF"
--   [4..7]   version: u32 (3)
--   [8..15]  tensor_count: u64
--   [16..23] metadata_kv_count: u64
--   [24..]   metadata tuples: (key, type, value)
--   [...]    tensor_infos: (name, n_dims, dims[], type, offset)
--   [align]  tensor_data (aligned to alignment bytes)
--
-- Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
---------------------------------------------------------------------

with Ada.Containers.Vectors;
with Ada.Strings.Unbounded;
with Interfaces;  -- for u32, u64
with System;       -- for System.Address

package LLM_GGUF is

   --------------------------------------------------------------------
   -- Types
   --------------------------------------------------------------------

   type U32 is new Interfaces.Unsigned_32;
   type U64 is new Interfaces.Unsigned_64;

   -- GGML quantized data types (subset relevant to Qwen Q5_K_M)
   type GGML_Type is
     (GGML_TYPE_F32,      -- 0: 32-bit float
      GGML_TYPE_F16,      -- 1: 16-bit float
      GGML_TYPE_Q4_0,     -- 2
      GGML_TYPE_Q4_1,     -- 3
      GGML_TYPE_Q5_0,     -- 6
      GGML_TYPE_Q5_1,     -- 7
      GGML_TYPE_Q8_0,     -- 8
      GGML_TYPE_Q8_1,     -- 9
      GGML_TYPE_Q2_K,     -- 10
      GGML_TYPE_Q3_K,     -- 11
      GGML_TYPE_Q4_K,     -- 12
      GGML_TYPE_Q5_K,     -- 13  ← Qwen 3.5 uses this
      GGML_TYPE_Q6_K,     -- 14
      GGML_TYPE_Q8_K);    -- 15

   type Dim_Array is array (1 .. 4) of U64;

   -- Tensor descriptor — where a tensor lives in the file
   type Tensor_Info is record
      Name       : Ada.Strings.Unbounded.Unbounded_String;
      N_Dims     : U32;
      Dims       : Dim_Array;
      Kind       : GGML_Type;
      Offset     : U64;  -- byte offset from file start (absolute within file)
   end record;

   package Tensor_Info_Vectors is new Ada.Containers.Vectors
     (Positive, Tensor_Info);

   -- Metadata key-value entry
   type Metadata_Entry is record
      Key   : Ada.Strings.Unbounded.Unbounded_String;
      Value : Ada.Strings.Unbounded.Unbounded_String;  -- all values as string
   end record;

   package Metadata_Vectors is new Ada.Containers.Vectors
     (Positive, Metadata_Entry);

   -- Complete GGUF file representation
   type GGUF_File is limited private;

   --------------------------------------------------------------------
   -- API
   --------------------------------------------------------------------

   -- Open and parse a GGUF file. Reads header + metadata + tensor infos.
   -- Does NOT load tensor data into memory — tensors are accessed via
   -- Read_Tensor_Raw on demand (memory-mapped style).
   procedure Open (File : out GGUF_File; Path : String)
     with Pre => Path'Length > 0;

   -- File status
   function Is_Open (File : GGUF_File) return Boolean;

   -- Total number of tensors in the file
   function Tensor_Count (File : GGUF_File) return Natural;

   -- Total number of metadata entries
   function Metadata_Count (File : GGUF_File) return Natural;

   -- Look up metadata by key. Returns empty string if not found.
   function Metadata (File : GGUF_File; Key : String) return String;

   -- Get tensor descriptor by index (1-based)
   function Tensor_At (File : GGUF_File; Index : Positive) return Tensor_Info
     with Pre => Index <= Tensor_Count (File);

   -- Find tensor by name. Raises Constraint_Error if not found.
   function Find_Tensor (File : GGUF_File; Name : String) return Tensor_Info;

   -- Read raw tensor data from file into byte array.
   -- Caller provides buffer large enough for the tensor.
   -- Size in bytes = n_elements × type_size
   -- type_size: F32=4, F16=2, Q5_K=block_aligned
   procedure Read_Tensor_Raw
     (File   : in out GGUF_File;
      Info   : Tensor_Info;
      Buffer : System.Address;
      Buf_Size : Natural);

   -- Byte size of a tensor given its type and element count
   function Tensor_Byte_Size (Info : Tensor_Info) return U64;

   -- Number of elements in a tensor (product of dims)
   function Tensor_Num_Elements (Info : Tensor_Info) return U64;

   -- Alignment used for tensor data in this file
   function Alignment (File : GGUF_File) return U64;

   -- Close the file
   procedure Close (File : in out GGUF_File);

   -- Quick helper: is this a valid GGUF file at path?
   function Is_GGUF (Path : String) return Boolean;

private

   type GGUF_File is record
      Is_Open       : Boolean := False;
      Path          : Ada.Strings.Unbounded.Unbounded_String;
      Version       : U32 := 0;
      Tensors       : Tensor_Info_Vectors.Vector;
      Meta          : Metadata_Vectors.Vector;
      Alignment_Val : U64 := 32;  -- default alignment
      Data_Start    : U64 := 0;   -- byte offset where tensor data begins
      FD            : Integer := -1;  -- POSIX file descriptor (via C import)
   end record;

end LLM_GGUF;
