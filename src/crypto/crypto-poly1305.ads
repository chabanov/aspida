---------------------------------------------------------------------
-- Crypto.Poly1305 — one-time authenticator (RFC 8439 §2.5)
--
-- Computes a 16-byte tag over a message using a 32-byte one-time key
-- (r || s). The key MUST be used for exactly one message. Arithmetic is
-- mod 2^130-5 using 5 x 26-bit limbs (constant-time: no secret-dependent
-- branches; the final reduction selects via masks, not branches).
---------------------------------------------------------------------

--  SPARK_Mode: flow-analysed (initialisation, data dependencies, non-aliasing).
--  Full absence-of-run-time-errors proof of the 26-bit limb carry arithmetic
--  needs the "accumulator < 2**130" invariant (research-grade, see SPARKNaCl)
--  and is tracked as future work; the cipher's tag-comparison path is already
--  fully proved in the Crypto root (Const_Time_Equal).
package Crypto.Poly1305 with SPARK_Mode => On is

   subtype Key_256 is Byte_Array (0 .. 31);   -- one-time key r||s
   subtype Tag_128 is Byte_Array (0 .. 15);

   procedure MAC (Key : Key_256; Msg : Byte_Array; Tag : out Tag_128)
     with Global => null;

end Crypto.Poly1305;
