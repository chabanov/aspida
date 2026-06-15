---------------------------------------------------------------------
-- Crypto.Poly1305 body — RFC 8439 §2.5 (5 x 26-bit limb implementation)
---------------------------------------------------------------------

with Interfaces; use Interfaces;

package body Crypto.Poly1305 with SPARK_Mode => On is

   Mask26 : constant U32 := 16#3FFFFFF#;

   procedure MAC (Key : Key_256; Msg : Byte_Array; Tag : out Tag_128) is
      --  Clamped r split into 26-bit limbs, and s_i = r_i * 5.
      R0 : constant U32 := Load_LE32 (Key, 0)                    and 16#3FFFFFF#;
      R1 : constant U32 := Shift_Right (Load_LE32 (Key, 3),  2)  and 16#3FFFF03#;
      R2 : constant U32 := Shift_Right (Load_LE32 (Key, 6),  4)  and 16#3FFC0FF#;
      R3 : constant U32 := Shift_Right (Load_LE32 (Key, 9),  6)  and 16#3F03FFF#;
      R4 : constant U32 := Shift_Right (Load_LE32 (Key, 12), 8)  and 16#00FFFFF#;
      S1 : constant U32 := R1 * 5;
      S2 : constant U32 := R2 * 5;
      S3 : constant U32 := R3 * 5;
      S4 : constant U32 := R4 * 5;

      --  Accumulator h (5 x 26-bit limbs).
      H0, H1, H2, H3, H4 : U32 := 0;

      N   : constant Natural := Msg'Length;
      Pos : Natural := 0;
   begin
      Tag := [others => 0];   -- fully initialised for flow; overwritten below
      while Pos < N loop
         declare
            Remain : constant Natural := N - Pos;
            Full   : constant Boolean := Remain >= 16;
            Take   : constant Natural := (if Full then 16 else Remain);
            Blk    : Byte_Array (0 .. 15) := [others => 0];
            Hibit  : U32;
            T0, T1, T2, T3 : U32;
            D0, D1, D2, D3, D4 : U64;
            C : U32;
         begin
            for I in 0 .. Take - 1 loop
               Blk (I) := Msg (Msg'First + Pos + I);
            end loop;
            if Full then
               Hibit := 16#1000000#;     -- implicit high bit: 1 << 24 in limb 4
            else
               Blk (Take) := 1;          -- append 0x01 right after the message
               Hibit := 0;
            end if;
            Pos := Pos + Take;

            --  h += block
            T0 := Load_LE32 (Blk, 0);
            T1 := Load_LE32 (Blk, 4);
            T2 := Load_LE32 (Blk, 8);
            T3 := Load_LE32 (Blk, 12);
            H0 := H0 + (T0 and Mask26);
            H1 := H1 + ((Shift_Right (T0, 26) or Shift_Left (T1, 6)) and Mask26);
            H2 := H2 + ((Shift_Right (T1, 20) or Shift_Left (T2, 12)) and Mask26);
            H3 := H3 + ((Shift_Right (T2, 14) or Shift_Left (T3, 18)) and Mask26);
            H4 := H4 + (Shift_Right (T3, 8) or Hibit);

            --  h *= r  (mod 2^130-5)
            D0 := U64 (H0) * U64 (R0) + U64 (H1) * U64 (S4) + U64 (H2) * U64 (S3)
                + U64 (H3) * U64 (S2) + U64 (H4) * U64 (S1);
            D1 := U64 (H0) * U64 (R1) + U64 (H1) * U64 (R0) + U64 (H2) * U64 (S4)
                + U64 (H3) * U64 (S3) + U64 (H4) * U64 (S2);
            D2 := U64 (H0) * U64 (R2) + U64 (H1) * U64 (R1) + U64 (H2) * U64 (R0)
                + U64 (H3) * U64 (S4) + U64 (H4) * U64 (S3);
            D3 := U64 (H0) * U64 (R3) + U64 (H1) * U64 (R2) + U64 (H2) * U64 (R1)
                + U64 (H3) * U64 (R0) + U64 (H4) * U64 (S4);
            D4 := U64 (H0) * U64 (R4) + U64 (H1) * U64 (R3) + U64 (H2) * U64 (R2)
                + U64 (H3) * U64 (R1) + U64 (H4) * U64 (R0);

            --  carry propagation (mask the low 26 bits in U64 BEFORE narrowing
            --  to U32 — the RFC C reference relies on truncating casts; Ada
            --  range-checks, so a 56-bit d would otherwise overflow U32).
            C  := U32 (Shift_Right (D0, 26)); H0 := U32 (D0 and 16#3FFFFFF#);
            D1 := D1 + U64 (C); C := U32 (Shift_Right (D1, 26)); H1 := U32 (D1 and 16#3FFFFFF#);
            D2 := D2 + U64 (C); C := U32 (Shift_Right (D2, 26)); H2 := U32 (D2 and 16#3FFFFFF#);
            D3 := D3 + U64 (C); C := U32 (Shift_Right (D3, 26)); H3 := U32 (D3 and 16#3FFFFFF#);
            D4 := D4 + U64 (C); C := U32 (Shift_Right (D4, 26)); H4 := U32 (D4 and 16#3FFFFFF#);
            H0 := H0 + C * 5;   C := Shift_Right (H0, 26); H0 := H0 and Mask26;
            H1 := H1 + C;
         end;
      end loop;

      --  Final full carry
      declare
         C : U32;
         G0, G1, G2, G3, G4 : U32;
         Mask : U32;
         F : U64;
         Pad0 : constant U32 := Load_LE32 (Key, 16);
         Pad1 : constant U32 := Load_LE32 (Key, 20);
         Pad2 : constant U32 := Load_LE32 (Key, 24);
         Pad3 : constant U32 := Load_LE32 (Key, 28);
      begin
         C := Shift_Right (H1, 26); H1 := H1 and Mask26;
         H2 := H2 + C; C := Shift_Right (H2, 26); H2 := H2 and Mask26;
         H3 := H3 + C; C := Shift_Right (H3, 26); H3 := H3 and Mask26;
         H4 := H4 + C; C := Shift_Right (H4, 26); H4 := H4 and Mask26;
         H0 := H0 + C * 5; C := Shift_Right (H0, 26); H0 := H0 and Mask26;
         H1 := H1 + C;

         --  compute h + -p  (g = h - p, via h + 5 then subtract 2^130)
         G0 := H0 + 5;  C := Shift_Right (G0, 26); G0 := G0 and Mask26;
         G1 := H1 + C;  C := Shift_Right (G1, 26); G1 := G1 and Mask26;
         G2 := H2 + C;  C := Shift_Right (G2, 26); G2 := G2 and Mask26;
         G3 := H3 + C;  C := Shift_Right (G3, 26); G3 := G3 and Mask26;
         G4 := H4 + C - 16#4000000#;   -- - (1 << 26); borrow sets bit 31

         --  constant-time select: h if h < p (borrow), else g (= h - p)
         Mask := Shift_Right (G4, 31) - 1;          -- 0 if borrow, else 0xFFFFFFFF
         G0 := G0 and Mask; G1 := G1 and Mask; G2 := G2 and Mask;
         G3 := G3 and Mask; G4 := G4 and Mask;
         Mask := not Mask;
         H0 := (H0 and Mask) or G0;
         H1 := (H1 and Mask) or G1;
         H2 := (H2 and Mask) or G2;
         H3 := (H3 and Mask) or G3;
         H4 := (H4 and Mask) or G4;

         --  collapse 5 x 26-bit limbs into 4 x 32-bit words
         H0 := H0 or Shift_Left (H1, 26);
         H1 := Shift_Right (H1, 6)  or Shift_Left (H2, 20);
         H2 := Shift_Right (H2, 12) or Shift_Left (H3, 14);
         H3 := Shift_Right (H3, 18) or Shift_Left (H4, 8);

         --  tag = (h + s) mod 2^128
         F := U64 (H0) + U64 (Pad0);                  H0 := U32 (F and 16#FFFFFFFF#);
         F := U64 (H1) + U64 (Pad1) + Shift_Right (F, 32); H1 := U32 (F and 16#FFFFFFFF#);
         F := U64 (H2) + U64 (Pad2) + Shift_Right (F, 32); H2 := U32 (F and 16#FFFFFFFF#);
         F := U64 (H3) + U64 (Pad3) + Shift_Right (F, 32); H3 := U32 (F and 16#FFFFFFFF#);

         Store_LE32 (Tag, 0,  H0);
         Store_LE32 (Tag, 4,  H1);
         Store_LE32 (Tag, 8,  H2);
         Store_LE32 (Tag, 12, H3);
      end;
   end MAC;

end Crypto.Poly1305;
