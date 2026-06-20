---------------------------------------------------------------------
-- Crypto.Memory — best-effort pinning of sensitive buffers in RAM so
-- secrets are never written to swap (mlock(2)). Returns False if the OS
-- declines (e.g. RLIMIT_MEMLOCK); callers treat it as advisory hardening,
-- not a guarantee. Note: pinning a multi-GB model is impractical, so this
-- is used only for small long-lived key material.
---------------------------------------------------------------------

with System;

package Crypto.Memory is

   --  Pin a buffer in RAM (mlock). Advisory hardening, not a guarantee.
   function Lock (First : System.Address; Length : Natural) return Boolean;

   --  Release a pin established by Lock (munlock). Symmetric counterpart so a
   --  caller that mlocks session keys on handshake can release them on Close.
   --  Best-effort: returns True on success, False if the OS declines.
   function Unlock (First : System.Address; Length : Natural) return Boolean;

end Crypto.Memory;
