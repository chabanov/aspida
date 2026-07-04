---------------------------------------------------------------------
-- LLM_Weight_Cache — client-side, content-addressed, AEAD-sealed chunk
-- cache for H19 weight-streaming (Phase 3 of H19_WEIGHT_STREAM_ROADMAP.md).
--
-- After the first session pulls a model over the channel, its weight chunks
-- are sealed at rest so a later session reads them locally — zero outbound
-- fetches, zero access-pattern leakage to the operator (the warm-cache
-- invariant). The seal reuses At_Rest (PBKDF2 + ChaCha20-Poly1305); the
-- per-model Cache_Key is bound into every blob's AEAD plaintext, so a blob
-- written under one key and read under another fails loud (Cache_Miss) —
-- never silent cross-model corruption.
--
-- Cache_Key today is an opaque string (the model id); Phase 4 will pass the
-- attested model HASH instead, so the key becomes trustworthy. Until then the
-- cache binds the key and the chunk index into every blob, but NOT the model's
-- content/version: if an operator republishes a *different* model under the
-- same id, a warm client can still unseal and serve the stale bytes (the key
-- matches). Fail-loud on a content change requires the attested hash — that is
-- the Phase-4 upgrade, and it is a key-derivation change only; the storage
-- layer below is unchanged by it.
--
-- Opt-in: a cache with an empty Dir or empty Pass is disabled (Has -> False,
-- Store -> no-op); the in-memory chunk cache in Remote_AEAD_Source still
-- works. Persistence is enabled by setting ASPIDA_WEIGHT_CACHE_DIR and
-- ASPIDA_WEIGHT_CACHE_PASS (read by LLM_Weight_Source.Open_Remote).
--
-- File layout: <Dir>/<Sanitized_Key>/chunk_<Index>.enc, where the sealed
-- blob is At_Rest's format wrapping plaintext = [Key_Len BE32][Key bytes]
-- [Index BE64][Chunk bytes]. Both the key and the chunk index are inside the
-- AEAD, so a blob read under the wrong key OR moved to a different index's
-- filename fails loud (Cache_Miss); a tampered file → Decrypt_Error →
-- Cache_Error.
---------------------------------------------------------------------

with Ada.Strings.Unbounded;
with Interfaces;
with Crypto;

package LLM_Weight_Cache is

   Cache_Error : exception;   --  format / I/O / tamper failure on a load
   Cache_Miss  : exception;   --  chunk absent, or present under a different key

   --  Shared heap-access to a byte buffer (the chunk cache and the weight
   --  source both hold chunk bytes this way).
   type Byte_Array_Access is access all Crypto.Byte_Array;

   --  Weight_Cache holds an owned password access, so it is not meant to be
   --  copied (a copy would share the password and double-wipe on Close). The
   --  public API exposes no function returning a Weight_Cache, so a caller
   --  cannot meaningfully copy one; it is non-limited only so it can be a
   --  by-value component of Remote_AEAD_Source (a non-limited tagged
   --  extension cannot have limited components).
   type Weight_Cache is private;

   --  Open an AEAD-sealed chunk cache rooted at Dir, scoped to Cache_Key. If
   --  Dir or Pass is empty the returned cache is disabled (Enabled -> False).
   --  On a non-empty Dir the per-model subdir <Dir>/<Sanitized(Cache_Key)> is
   --  created (with parents) so At_Rest.Save can write into it. The returned
   --  Weight_Cache owns its password copy (wiped on Close).
   function Open
     (Dir       : String;
      Pass      : Crypto.Byte_Array;
      Cache_Key : String) return Weight_Cache;

   function Enabled (C : Weight_Cache) return Boolean;

   --  True iff a sealed blob for chunk Index exists on disk. Always False when
   --  disabled.
   function Has
     (C     : Weight_Cache;
      Index : Interfaces.Unsigned_64) return Boolean;

   --  Load+unseal chunk Index. Raises Cache_Miss if the blob is absent or was
   --  written under a different Cache_Key; Cache_Error on tamper / format /
   --  I/O error. Returns a heap-allocated 0-based Byte_Array (caller owns it).
   function Load
     (C     : Weight_Cache;
      Index : Interfaces.Unsigned_64) return Byte_Array_Access;

   --  Seal+atomically write chunk Index. No-op when disabled.
   procedure Store
     (C     : in out Weight_Cache;
      Index : Interfaces.Unsigned_64;
      Data  : Crypto.Byte_Array);

   --  Wipe the held password and release state. Idempotent.
   procedure Close (C : in out Weight_Cache);

private

   type Weight_Cache is record
      On   : Boolean := False;
      Dir  : Ada.Strings.Unbounded.Unbounded_String;   --  cache root
      Key  : Ada.Strings.Unbounded.Unbounded_String;   --  bound into each blob
      Pass : Byte_Array_Access;                          --  owned; wiped on Close
   end record;

end LLM_Weight_Cache;