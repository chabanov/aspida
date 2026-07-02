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
with Interfaces;       -- for u32, u64
with LLM_Byte_Source;  -- random-access byte source (local file today, remote H19 later)
with System;           -- for System.Address

package LLM_GGUF is

   --  `=` / `/=` on the byte-source access (used by Open_From_Source's Pre).
   use type LLM_Byte_Source.Byte_Source_Access;

   --  Raised when an untrusted GGUF file is structurally invalid: a tensor
   --  whose declared size overflows or runs past the end of the file, an
   --  out-of-range value-type / dimension count, a bad alignment, etc.
   --  Failing loudly here (rather than reading out of bounds or dividing by
   --  zero) keeps a hostile model from corrupting the loader.
   Malformed_GGUF : exception;

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
      GGML_TYPE_Q8_K,     -- 15
      GGML_TYPE_BF16,     -- 30 (brain float; Gemma per-layer projection)
      GGML_TYPE_UNKNOWN); -- any ggml type we do not implement (IQ*, ternary,
                           --  legacy): never decode as something else — load
                           --  must reject it loudly, not silently zero/garbage.

   type Dim_Array is array (1 .. 4) of U64;

   -- Tensor descriptor — where a tensor lives in the file
   type Tensor_Info is record
      Name       : Ada.Strings.Unbounded.Unbounded_String;
      N_Dims     : U32;
      Dims       : Dim_Array;
      Kind       : GGML_Type;
      Offset     : U64;  -- byte offset from file start (absolute within file)
      --  Validated, overflow-checked size in bytes of this tensor's data,
      --  computed once in Open from the (untrusted) dims + type and bounded
      --  against the file length. The single source of truth consumed by the
      --  decode loop count and every allocation; see Tensor_Byte_Size.
      Byte_Size  : U64 := 0;
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

   --  H19 Phase 7 partial-warm: a heap-allocated GGUF_File the model can keep
   --  alive for a background block-fetcher to stream layers K+1..N. Shared by
   --  the engine (allocates), the backend (passes through), and the model
   --  (holds + frees). Visible because GGUF_File is opaque (limited private):
   --  only this unit can create/destroy the record, but callers hold access
   --  values to keep one alive past the load call.
   type GGUF_Ptr is access all GGUF_File;

   --------------------------------------------------------------------
   -- API
   --------------------------------------------------------------------

   -- Open and parse a GGUF file. Reads header + metadata + tensor infos.
   -- Does NOT load tensor data into memory — tensors are accessed via
   -- Read_Tensor_Raw on demand (memory-mapped style).
   procedure Open (File : out GGUF_File; Path : String)
     with Pre => Path'Length > 0;

   --  H19 (weight-streaming): parse a GGUF from an ALREADY-OPEN byte source
   --  (a Remote_AEAD_Source over the Secure_Channel, or any Byte_Source).
   --  Takes ownership of Src: on success the GGUF_File owns it (Close frees
   --  it); on a parse failure Src is freed and Malformed_GGUF is raised; on a
   --  bad-magic / absurd-count header Src is freed and File.Is_Open is left
   --  False (matching Open's silent-return for a non-GGUF / hostile header).
   --  This is the seam that lets the engine load a model whose bytes arrive
   --  over the encrypted channel instead of from a local file, with no change
   --  to the parser or any backend.
   procedure Open_From_Source (File : out GGUF_File; Src : LLM_Byte_Source.Byte_Source_Access)
     with Pre => Src /= null;

   -- File status
   function Is_Open (File : GGUF_File) return Boolean;

   -- Total number of tensors in the file
   function Tensor_Count (File : GGUF_File) return Natural;

   -- Total number of metadata entries
   function Metadata_Count (File : GGUF_File) return Natural;

   -- Look up metadata by key. Returns empty string if not found.
   function Metadata (File : GGUF_File; Key : String) return String;

   -- Iterate metadata entries by 1-based index (for dumping/inspection).
   function Meta_Key_At   (File : GGUF_File; Index : Positive) return String;
   function Meta_Value_At (File : GGUF_File; Index : Positive) return String;

   -- Get tensor descriptor by index (1-based)
   function Tensor_At (File : GGUF_File; Index : Positive) return Tensor_Info
     with Pre => Index <= Tensor_Count (File);

   -- Find tensor by name. Raises Constraint_Error if not found.
   function Find_Tensor (File : GGUF_File; Name : String) return Tensor_Info;

   -- True iff a tensor with this exact name exists. Lets callers probe for
   -- optional tensors (e.g. a shared expert that some MoE variants omit)
   -- without the Constraint_Error that Find_Tensor raises on a miss.
   function Has_Tensor (File : GGUF_File; Name : String) return Boolean;

   -- Read raw tensor data from file into byte array.
   -- Caller provides buffer large enough for the tensor.
   -- Size in bytes = n_elements × type_size
   -- type_size: F32=4, F16=2, Q5_K=block_aligned
   procedure Read_Tensor_Raw
     (File   : in out GGUF_File;
      Info   : Tensor_Info;
      Buffer : System.Address;
      Buf_Size : Natural);

   --  Read Buf_Size bytes starting Byte_Offset into a tensor's data — used to
   --  stream individual rows of tensors too large to hold in one buffer
   --  (e.g. a >2 GiB per-layer embedding table). The file is kept open.
   procedure Read_Tensor_Range
     (File        : in out GGUF_File;
      Info        : Tensor_Info;
      Byte_Offset : U64;
      Buffer      : System.Address;
      Buf_Size    : Natural);

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

   --------------------------------------------------------------------
   -- Tokenizer arrays (captured structurally during Open so that token
   -- strings containing commas/brackets are preserved, unlike the
   -- comma-joined Metadata string form).
   --------------------------------------------------------------------

   -- Number of vocabulary tokens (tokenizer.ggml.tokens), 0 if absent.
   function Token_Count (File : GGUF_File) return Natural;
   -- Vocabulary token by 1-based index (id = Index - 1).
   function Token_At (File : GGUF_File; Index : Positive) return String;

   -- Number of BPE merge rules (tokenizer.ggml.merges), 0 if absent.
   function Merge_Count (File : GGUF_File) return Natural;
   -- Merge rule by 1-based index, in the GGUF "left right" form.
   function Merge_At (File : GGUF_File; Index : Positive) return String;

private

   package Str_Vectors is new Ada.Containers.Vectors
     (Positive, Ada.Strings.Unbounded.Unbounded_String,
      Ada.Strings.Unbounded."=");

   type GGUF_File is record
      Is_Open       : Boolean := False;
      Path          : Ada.Strings.Unbounded.Unbounded_String;
      Version       : U32 := 0;
      Tensors       : Tensor_Info_Vectors.Vector;
      Meta          : Metadata_Vectors.Vector;
      Tokens        : Str_Vectors.Vector;  -- tokenizer.ggml.tokens
      Merges        : Str_Vectors.Vector;  -- tokenizer.ggml.merges
      Alignment_Val : U64 := 32;  -- default alignment
      Data_Start    : U64 := 0;   -- byte offset where tensor data begins
      File_Size     : U64 := 0;   -- total byte length (from the Byte_Source)
      --  The byte source behind on-demand tensor reads. Today a
      --  Local_File_Source (POSIX fd); H19 will swap in a Remote_AEAD_Source
      --  (byte-range fetch over the Secure_Channel + local cache) with no
      --  change to the parser or any backend. Owned by the GGUF_File: Close
      --  frees it (closes the fd / connection and deallocates the access).
      Source        : LLM_Byte_Source.Byte_Source_Access := null;
   end record;

end LLM_GGUF;
