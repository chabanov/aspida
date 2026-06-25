------------------------------------------------------------------------
-- Platform_Auth — authorize an engineer by their UARP API key. The engineer
-- enters their `uarp_<prefix>_<secret>` key; we validate it against the UARP
-- backend (GET /api/v1/me) and, on success, get their user id for ownership /
-- billing. One key works across snaga.ai (UARP) and the training platform.
--
-- The engine has no TLS client, so verification shells out to `curl` (like
-- Exec_Verifier -> python3); the key is passed via a 0600 curl config file, not
-- argv. Verified keys are cached for the process to avoid re-calling UARP.
--
-- Base URL from ASPIDA_UARP_URL (default https://snaga.ai).
------------------------------------------------------------------------

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Platform_Auth is

   --  Validate an engineer's UARP API key. Ok = key accepted by UARP; Identity =
   --  their UARP tenant id (the auth unit for api keys; /me returns user=null,
   --  tenant.tenant_id for key auth). Empty if not Ok. Available = curl present.
   procedure Verify
     (Key : String; Ok : out Boolean; Identity : out Unbounded_String);

   --  curl available (else verification can't run).
   function Available return Boolean;

end Platform_Auth;
