---------------------------------------------------------------------
-- Crypto.X25519 body — RFC 7748.
--
-- Field elements use 16 signed limbs of ~16 bits (the TweetNaCl "gf"
-- representation); car25519 renormalises. Ported faithfully from the
-- TweetNaCl reference; the field operations are expressed as Ada
-- functions returning fresh GF values (no aliasing of out-parameters),
-- and C's signed bitwise idioms become an arithmetic-shift helper plus a
-- two's-complement Unchecked_Conversion mask for the constant-time swap.
---------------------------------------------------------------------

with Interfaces;             use Interfaces;
with Ada.Unchecked_Conversion;

package body Crypto.X25519 with SPARK_Mode => On is

   subtype I64 is Interfaces.Integer_64;
   type GF is array (0 .. 15) of I64;

   function To_U is new Ada.Unchecked_Conversion (I64, U64);
   function To_I is new Ada.Unchecked_Conversion (U64, I64);

   C121665 : constant GF := [0 => 16#DB41#, 1 => 1, others => 0];

   --  Arithmetic shift right by 16 = floor(X / 2^16) (C's signed >> 16).
   function Asr16 (X : I64) return I64 is
      Q : constant I64 := X / 65536;
   begin
      if X < 0 and then Q * 65536 /= X then
         return Q - 1;
      else
         return Q;
      end if;
   end Asr16;

   procedure Car (O : in out GF) is
      C : I64;
   begin
      for I in 0 .. 15 loop
         O (I) := O (I) + 65536;            -- + (1 << 16)
         C := Asr16 (O (I));
         if I < 15 then
            O (I + 1) := O (I + 1) + (C - 1);
         else
            O (0) := O (0) + 38 * (C - 1);  -- wrap: 2^256 = 38 (mod p)
         end if;
         O (I) := O (I) - C * 65536;        -- - (c << 16)
      end loop;
   end Car;

   --  Constant-time conditional swap of P and Q when B = 1.
   procedure Sel (P, Q : in out GF; B : I64) is
      Mask : constant U64 := (if B /= 0 then 16#FFFFFFFFFFFFFFFF# else 0);
      T    : I64;
   begin
      for I in 0 .. 15 loop
         T := To_I (Mask and (To_U (P (I)) xor To_U (Q (I))));
         P (I) := To_I (To_U (P (I)) xor To_U (T));
         Q (I) := To_I (To_U (Q (I)) xor To_U (T));
      end loop;
   end Sel;

   function Add (A, B : GF) return GF is
      O : GF := [others => 0];
   begin
      for I in 0 .. 15 loop O (I) := A (I) + B (I); end loop;
      return O;
   end Add;

   function Sub (A, B : GF) return GF is
      O : GF := [others => 0];
   begin
      for I in 0 .. 15 loop O (I) := A (I) - B (I); end loop;
      return O;
   end Sub;

   function Mul (A, B : GF) return GF is
      T : array (0 .. 30) of I64 := [others => 0];
      O : GF := [others => 0];
   begin
      for I in 0 .. 15 loop
         for J in 0 .. 15 loop
            T (I + J) := T (I + J) + A (I) * B (J);
         end loop;
      end loop;
      for I in 0 .. 14 loop
         T (I) := T (I) + 38 * T (I + 16);   -- fold high half: 2^256 = 38
      end loop;
      for I in 0 .. 15 loop O (I) := T (I); end loop;
      Car (O); Car (O);
      return O;
   end Mul;

   function Sqr (A : GF) return GF is (Mul (A, A));

   --  i^(p-2) = i^-1 (mod p), via the fixed 255-step addition chain.
   function Inv (Inp : GF) return GF is
      C : GF := Inp;
   begin
      for A in reverse 0 .. 253 loop
         C := Sqr (C);
         if A /= 2 and then A /= 4 then
            C := Mul (C, Inp);
         end if;
      end loop;
      return C;
   end Inv;

   function Unpack (N : Key_256) return GF is
      O : GF := [others => 0];
   begin
      for I in 0 .. 15 loop
         O (I) := I64 (N (N'First + 2 * I))
                + I64 (N (N'First + 2 * I + 1)) * 256;
      end loop;
      O (15) := O (15) mod 16#8000#;          -- clear bit 255 (& 0x7fff)
      return O;
   end Unpack;

   procedure Pack (O : out Key_256; N : GF) is
      M : GF := [others => 0];
      T : GF := [others => 0];
      B : I64;
   begin
      O := [others => 0];
      T := N;
      Car (T); Car (T); Car (T);
      for J in 0 .. 1 loop
         M (0) := T (0) - 16#FFED#;
         for I in 1 .. 14 loop
            M (I) := T (I) - 16#FFFF# - (if M (I - 1) < 0 then 1 else 0);
            M (I - 1) := M (I - 1) mod 16#10000#;    -- & 0xffff
         end loop;
         M (15) := T (15) - 16#7FFF# - (if M (14) < 0 then 1 else 0);
         B := (if M (15) < 0 then 1 else 0);
         M (14) := M (14) mod 16#10000#;
         Sel (T, M, 1 - B);
      end loop;
      for I in 0 .. 15 loop
         O (2 * I)     := U8 (T (I) mod 256);
         O (2 * I + 1) := U8 ((T (I) / 256) mod 256);
      end loop;
   end Pack;

   function Scalar_Mult (Scalar, Point : Key_256) return Key_256 is
      Z : Byte_Array (0 .. 31) := [others => 0];
      X : GF := [others => 0];
      A : GF := [0 => 1, others => 0];
      B : GF := [others => 0];
      C : GF := [others => 0];
      D : GF := [0 => 1, others => 0];
      E : GF := [others => 0];
      F : GF := [others => 0];
      R : I64;
      Result : Key_256 := [others => 0];
   begin
      --  Clamp the scalar (RFC 7748 §5).
      for I in 0 .. 30 loop
         Z (I) := Scalar (Scalar'First + I);
      end loop;
      Z (31) := (Scalar (Scalar'First + 31) and 127) or 64;
      Z (0)  := Z (0) and 248;

      X := Unpack (Point);
      B := X;

      for I in reverse 0 .. 254 loop
         R := I64 (Shift_Right (Z (I / 8), Natural (I mod 8)) and 1);
         Sel (A, B, R);
         Sel (C, D, R);
         E := Add (A, C);
         A := Sub (A, C);
         C := Add (B, D);
         B := Sub (B, D);
         D := Sqr (E);
         F := Sqr (A);
         A := Mul (C, A);
         C := Mul (B, E);
         E := Add (A, C);
         A := Sub (A, C);
         B := Sqr (A);
         C := Sub (D, F);
         A := Mul (C, C121665);
         A := Add (A, D);
         C := Mul (C, A);
         A := Mul (D, F);
         D := Mul (B, X);
         B := Sqr (E);
         Sel (A, B, R);
         Sel (C, D, R);
      end loop;

      C := Inv (C);
      A := Mul (A, C);
      Pack (Result, A);
      Wipe (Z);   -- scrub the clamped private scalar before returning
      return Result;
   end Scalar_Mult;

   function Public_Key (Scalar : Key_256) return Key_256 is
      Base : constant Key_256 := [0 => 9, others => 0];
   begin
      return Scalar_Mult (Scalar, Base);
   end Public_Key;

end Crypto.X25519;
