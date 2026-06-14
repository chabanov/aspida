---------------------------------------------------------------------
-- Crypto.Memory — best-effort pinning of sensitive buffers in RAM so
-- secrets are never written to swap (mlock(2)). Returns False if the OS
-- declines (e.g. RLIMIT_MEMLOCK); callers treat it as advisory hardening,
-- not a guarantee. Note: pinning a multi-GB model is impractical, so this
-- is used only for small long-lived key material.
---------------------------------------------------------------------

with System;

package Crypto.Memory is

   function Lock (First : System.Address; Length : Natural) return Boolean;

end Crypto.Memory;
