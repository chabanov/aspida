---------------------------------------------------------------------
-- Crypto.X25519 — Diffie-Hellman on Curve25519 (RFC 7748).
--
-- Scalar_Mult computes X25519(scalar, u): the Montgomery ladder over
-- GF(2^255-19), constant-time (fixed iteration count, conditional swaps
-- via masks). Scalars are clamped per the RFC. Public_Key is the special
-- case with the base point u = 9. A shared secret is
--   Scalar_Mult (my_secret, their_public).
---------------------------------------------------------------------

--  SPARK_Mode: flow-analysed (initialisation, data dependencies, non-aliasing).
--  Full absence-of-run-time-errors proof of the 16-limb field arithmetic needs
--  limb-bound invariants (research-grade, like Poly1305) and is future work.
package Crypto.X25519 with SPARK_Mode => On is

   subtype Key_256 is Byte_Array (0 .. 31);

   function Scalar_Mult (Scalar, Point : Key_256) return Key_256
     with Global => null;

   function Public_Key (Scalar : Key_256) return Key_256
     with Global => null;

end Crypto.X25519;
