---------------------------------------------------------------------
-- LLM_Weight_Pin — attestation-pinned model hash (H19 Phase 4 / H17).
--
-- The client-side integrity check for streamed weights. The operator pins a
-- model to its SHA-256 digest out-of-band (a release channel, a release
-- note); the client loads that pin from ASPIDA_WEIGHT_PIN and verifies the
-- bytes it streams from the (untrusted) weight store hash to it before it
-- trusts them for inference. A mismatched, swapped, or tampered model fails
-- loud (Pin_Error) — never silent corruption.
--
-- Pin format (ASPIDA_WEIGHT_PIN):
--   <model_id>@<hex>   -- 64 hex chars (SHA-256); model_id binds the pin to a
--                          specific model so a swap to a different pinned
--                          model is still caught
--   <hex>              -- bare 64-hex digest, no model-id binding
--   (empty)            -- no pin: Verify is a no-op (the Phase 1-3 behavior,
--                          unchanged — integrity is the AEAD channel alone)
--
-- Signed-manifest / Ed25519 attestation is a deliberate follow-up: the crypto
-- layer has no signature primitive yet, so this ships the pinned-hash half
-- now (the part that needs only SHA-256, which we have and prove) and stages
-- the signing half. The pin is the trust root today; the manifest will later
-- make the pin itself attestable.
--
-- Hash_Source streams a whole Byte_Source through the incremental SHA-256
-- Context (Init -> Update* -> Final), so a model larger than memory is hashed
-- without materializing it. The Context is cross-checked bit-identical to the
-- one-shot Hash on boundary inputs in test_weight_pin.
---------------------------------------------------------------------

with Ada.Strings.Unbounded;
with Crypto.SHA256;
with LLM_Byte_Source;

package LLM_Weight_Pin is

   --  Raised by Parse on a malformed pin spec (wrong length, non-hex), and by
   --  Verify on a model-id or digest mismatch.
   Pin_Error : exception;

   ------------------------------------------------------------------
   -- Pin — a parsed ASPIDA_WEIGHT_PIN
   ------------------------------------------------------------------

   type Pin is private;

   --  The empty pin (no verification). Verify against it is a no-op.
   Empty_Pin : constant Pin;

   --  Parse a pin spec. "" => Empty_Pin. "<id>@<hex>" or "<hex>" (64 hex).
   --  Raises Pin_Error on a malformed spec (hex length /= 64, non-hex char,
   --  empty id before '@', empty hex after '@').
   function Parse (Spec : String) return Pin;

   function Is_Empty (P : Pin) return Boolean;

   --  The bound model id, or "" if the pin is a bare hash (or empty).
   function Pin_Model_ID (P : Pin) return String;

   --  The pinned digest. Empty_Pin's digest is all-zero (never matches a real
   --  model, but Is_Empty gates Verify so it is never compared).
   function Pin_Digest (P : Pin) return Crypto.SHA256.Digest;

   ------------------------------------------------------------------
   -- Hashing + verification against a Byte_Source
   ------------------------------------------------------------------

   --  SHA-256 of the entire source: Seek (0), stream in fixed-size chunks
   --  through the incremental Context, Final. The source cursor is left at
   --  Byte_Length (end). Raises LLM_Byte_Source.Malformed_Source on a short
   --  read (a truncated source).
   function Hash_Source
     (S : in out LLM_Byte_Source.Byte_Source'Class)
      return Crypto.SHA256.Digest;

   --  Verify S against P. If P is empty, this is a no-op (Phase 1-3 behavior:
   --  integrity is the AEAD channel alone). Otherwise:
   --    * if P binds a model id and it differs from Model_ID, raise Pin_Error
   --      ("pin model-id mismatch") — catches a swap to a different pinned
   --      model without hashing;
   --    * hash S and compare to P's digest; on mismatch raise Pin_Error
   --      ("weight hash mismatch") — catches a tampered or swapped model.
   --  Model_ID is the id the caller opened the source with (e.g. the
   --  Remote_AEAD_Source's Model_ID); it is matched against the pin's id.
   procedure Verify
     (S        : in out LLM_Byte_Source.Byte_Source'Class;
      Model_ID : String;
      P        : Pin);

private

   type Pin is record
      Has_ID : Boolean := False;
      ID     : Ada.Strings.Unbounded.Unbounded_String :=
                 Ada.Strings.Unbounded.Null_Unbounded_String;
      Dig    : Crypto.SHA256.Digest := [others => 0];
   end record;

   Empty_Pin : constant Pin :=
     (Has_ID => False,
      ID     => Ada.Strings.Unbounded.Null_Unbounded_String,
      Dig    => [others => 0]);

end LLM_Weight_Pin;