---------------------------------------------------------------------
-- Crypto.PBKDF2 — password-based key derivation (PBKDF2-HMAC-SHA256,
-- RFC 8018 §5.2). Stretches a password + salt into key material over a
-- configurable iteration count (work factor) to slow brute force.
---------------------------------------------------------------------

--  SPARK_Mode: flow-analysed (initialisation, data dependencies, non-aliasing)
--  and, since the index arithmetic was given loop invariants, fully proved
--  (AoRTE) at --mode=all.
package Crypto.PBKDF2 with SPARK_Mode => On is

   --  Salt is a small fixed-size random value (we use 16 bytes); cap it well
   --  above any sane use so the per-block message buffer Salt'Length + 4 stays
   --  trivially within a Natural index. DK is bounded by the RFC's 255*32 limit.
   Max_Salt_Len : constant := 256;

   procedure Derive
     (Password   : Byte_Array;
      Salt       : Byte_Array;
      Iterations : Positive;
      DK         : out Byte_Array)
     with Global => null,
          Pre    => Salt'Length <= Max_Salt_Len
                    and then DK'Length <= 255 * 32;

end Crypto.PBKDF2;
