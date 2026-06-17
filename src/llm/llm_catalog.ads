---------------------------------------------------------------------
-- LLM_Catalog — enumerate the GGUF models present on this system.
--
-- Walks a set of search roots (ASPIDA_MODELS_DIR plus common LLM model
-- directories), finds every *.gguf, and classifies each WITHOUT loading any
-- weights: it reads only the GGUF metadata (architecture, name, size label)
-- and the file size. Multimodal projector files (mmproj-*) are flagged, not
-- treated as runnable models. "Supported" is decided by the engine's own
-- backend registry (LLM_Engine.Supports), so the catalog never drifts from
-- what can actually be loaded.
--
-- Intended use: call Discover once at startup so the inference server "sees"
-- all available models and a client can pick one. Cheap and read-only.
---------------------------------------------------------------------

with Ada.Strings.Unbounded;
with Ada.Containers.Vectors;

package LLM_Catalog is

   package SU renames Ada.Strings.Unbounded;

   type Model_Status is
     (Supported,    -- valid GGUF, architecture the engine can run
      Unsupported,  -- valid GGUF, architecture not in the engine registry
      Projector,    -- multimodal projector (mmproj-*): not a standalone LLM
      Invalid);     -- not a readable GGUF

   type Model_Entry is record
      Path   : SU.Unbounded_String;          -- absolute path on disk
      Name   : SU.Unbounded_String;          -- general.name, else file basename
      Arch   : SU.Unbounded_String;          -- general.architecture ("" if none)
      Quant  : SU.Unbounded_String;          -- quant tag from the name (e.g. Q4_K_M)
      Params : SU.Unbounded_String;          -- general.size_label (e.g. 70B), else ""
      Size   : Long_Long_Integer := 0;       -- file size in bytes
      Status : Model_Status := Invalid;
   end record;

   package Entry_Vectors is new Ada.Containers.Vectors (Positive, Model_Entry);

   --  Scan every search root for *.gguf and classify each (metadata only, no
   --  weights). Results are sorted Supported-first, then by name. De-duplicated
   --  by absolute path. Safe at startup; never raises (unreadable dirs/files
   --  are skipped).
   function Discover return Entry_Vectors.Vector;

   --  The search roots that exist on this host, in scan order, ':'-joined
   --  (for logging).
   function Roots_Description return String;

   --  One human-readable line for an entry (status, name, arch, quant, size).
   function Describe (E : Model_Entry) return String;

   --  Pretty byte size, e.g. "40.0 GB", "4.6 GB", "812 MB".
   function Human_Size (Bytes : Long_Long_Integer) return String;

end LLM_Catalog;
