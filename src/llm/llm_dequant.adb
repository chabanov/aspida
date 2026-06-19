---------------------------------------------------------------------
-- LLM_Dequant body — GGML quantization → FP32 conversion
---------------------------------------------------------------------

with Interfaces; use Interfaces;
with LLM_Pool;

package body LLM_Dequant is

   --  This unit is the quantized-matvec hot path: every element is reached by
   --  index arithmetic that is correct by construction (super-block layout is
   --  fixed, ranges validated bit-exact against llama + the real-model tests).
   --  Suppressing the per-element index/range/overflow checks lets the decode
   --  and dot inner loops run without that tax (and helps the compiler keep
   --  them in registers / vectorise). Checks stay on everywhere else.
   pragma Suppress (All_Checks);

   use LLM_Tensor;

   subtype Byte is Interfaces.Unsigned_8;

   --  One dequantized super-block (256 elements) on the stack — lets the hot
   --  matvec kernel fuse decode + dot over local arrays (vectorisable, no heap,
   --  no Tensor access indirection).
   type Block256 is array (1 .. 256) of Float;

   --------------------------------------------------------------------
   -- F16 → F32 conversion
   --------------------------------------------------------------------

   --  Decode an IEEE-754 half from its two little-endian bytes (Lo = byte 0,
   --  Hi = byte 1). Sign/exponent/mantissa-high live in the HIGH byte.
   function F16_To_F32 (Lo : Byte; Hi : Byte) return Float is
      Sign     : constant Integer := Integer (Hi and 128);
      Exponent : constant Integer := Integer (Shift_Right (Hi and 124, 2));
      Mantissa : constant Integer :=
        Integer (Shift_Left (Unsigned_16 (Hi and 3), 8)) + Integer (Lo);
      Value    : Float;
   begin
      if Exponent = 0 then
         -- Subnormal or zero
         if Mantissa = 0 then
            return (if Sign = 0 then 0.0 else -0.0);
         end if;
         Value := Float (Mantissa) / 1024.0 * 2.0 ** (-14);
      elsif Exponent = 31 then
         -- Infinity or NaN
         return 0.0;
      else
         Value := Float (Mantissa + 1024) / 1024.0 * 2.0 ** (Exponent - 15);
      end if;

      if Sign /= 0 then
         Value := -Value;
      end if;
      return Value;
   end F16_To_F32;

   --------------------------------------------------------------------
   -- Q8_K dequantization
   --
   -- Super-block of 256 elements:
   --   - 2 bytes: FP16 d (scale for entire block)
   --   - 256 bytes: qs (int8 values, unsigned)
   -- Formula: result[i] = d * qs[i]
   -- Total per 256 el: 2 + 256 = 258 bytes
   --------------------------------------------------------------------

   --  Q8_K super-block (llama.cpp block_q8_K, 292 bytes / 256 elements):
   --    float   d;            -- 4-byte little-endian block scale
   --    int8_t  qs[256];      -- signed quants
   --    int16_t bsums[16];    -- partial sums (unused for plain dequant)
   --  Value: y[i] = d * qs[i].
   procedure Dequant_Q8_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;

      function Read_F32 (At_Pos : Natural) return Float is
         subtype Bytes4 is String (1 .. 4);
         Buf : Bytes4;
         F   : Float;
         for F'Address use Buf'Address;
      begin
         Buf := X (At_Pos .. At_Pos + 3);
         return F;
      end Read_F32;
   begin
      for B in 1 .. Blocks loop
         declare
            D : constant Float := Read_F32 (Pos);   -- f32 scale (4 bytes)
         begin
            Pos := Pos + 4;
            for I in 1 .. 256 loop
               declare
                  U : constant Integer := Character'Pos (X (Pos + I - 1));
                  S : constant Integer := (if U >= 128 then U - 256 else U); -- int8
               begin
                  Set_Flat (Q, Q_Pos, D * Float (S));
                  Q_Pos := Q_Pos + 1;
               end;
            end loop;
            Pos := Pos + 256;   -- qs
            Pos := Pos + 32;    -- skip bsums (16 x int16)
         end;
      end loop;
   end Dequant_Q8_K;

   --------------------------------------------------------------------
   -- Q6_K dequantization
   --
   -- Super-block (llama.cpp block_q6_K, 210 bytes / 256 elements):
   --   uint8_t ql[128];   -- lower 4 bits of each quant
   --   uint8_t qh[64];    -- upper 2 bits of each quant
   --   int8_t  scales[16];-- per-16-element signed scales
   --   ggml_half d;       -- f16 super-block scale (last)
   --  Each 256-block is two 128-element halves; per half ql/qh/scales
   --  advance by 64/32/8. Value: y = d * scales[is] * (q6 - 32).
   --------------------------------------------------------------------

   --  Decode ONE Q6_K super-block (210 bytes at Pos) into B (1..256).
   procedure Decode_Q6K_Block (X : String; Pos : Natural; B : out Block256) is
      function S8 (P : Natural) return Integer is
         U : constant Integer := Character'Pos (X (P));
      begin
         return (if U >= 128 then U - 256 else U);
      end S8;
      QL_Base : constant Natural := Pos;
      QH_Base : constant Natural := Pos + 128;
      SC_Base : constant Natural := Pos + 192;
      D       : constant Float := F16_To_F32 (
        Byte (Character'Pos (X (Pos + 208))),
        Byte (Character'Pos (X (Pos + 209))));
   begin
      for Half in 0 .. 1 loop
         declare
            QL_H : constant Natural := QL_Base + Half * 64;
            QH_H : constant Natural := QH_Base + Half * 32;
            SC_H : constant Natural := SC_Base + Half * 8;
            Y_H  : constant Natural := Half * 128;   -- block-local base
         begin
            --  Each half splits into two 16-element groups (Is = L/16) that
            --  share four scales. Hoist the scale lookups out of the element
            --  loop (no per-element S8 call or division) so the branchless
            --  extraction + scale of all four quant streams vectorises.
            for IG in 0 .. 1 loop
               declare
                  DS1 : constant Float := D * Float (S8 (SC_H + IG + 0));
                  DS2 : constant Float := D * Float (S8 (SC_H + IG + 2));
                  DS3 : constant Float := D * Float (S8 (SC_H + IG + 4));
                  DS4 : constant Float := D * Float (S8 (SC_H + IG + 6));
                  LB  : constant Natural := IG * 16;
               begin
            for L in LB .. LB + 15 loop
               declare
                  QL_L   : constant Unsigned_32 :=
                    Unsigned_32 (Character'Pos (X (QL_H + L)));
                  QL_L32 : constant Unsigned_32 :=
                    Unsigned_32 (Character'Pos (X (QL_H + L + 32)));
                  QH_L   : constant Unsigned_32 :=
                    Unsigned_32 (Character'Pos (X (QH_H + L)));
                  Q1 : constant Integer :=
                    Integer ((QL_L and 16#0F#) or Shift_Left (QH_L and 3, 4)) - 32;
                  Q2 : constant Integer :=
                    Integer ((QL_L32 and 16#0F#)
                             or Shift_Left (Shift_Right (QH_L, 2) and 3, 4)) - 32;
                  Q3 : constant Integer :=
                    Integer (Shift_Right (QL_L, 4)
                             or Shift_Left (Shift_Right (QH_L, 4) and 3, 4)) - 32;
                  Q4 : constant Integer :=
                    Integer (Shift_Right (QL_L32, 4)
                             or Shift_Left (Shift_Right (QH_L, 6) and 3, 4)) - 32;
               begin
                  B (Y_H + L + 1)  := DS1 * Float (Q1);
                  B (Y_H + L + 33) := DS2 * Float (Q2);
                  B (Y_H + L + 65) := DS3 * Float (Q3);
                  B (Y_H + L + 97) := DS4 * Float (Q4);
               end;
            end loop;
               end;
            end loop;
         end;
      end loop;
   end Decode_Q6K_Block;

   procedure Dequant_Q6_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Bk     : Block256;
      QP     : Natural := 1;
   begin
      for Blk in 0 .. Blocks - 1 loop
         Decode_Q6K_Block (X, X'First + Blk * 210, Bk);
         for I in 1 .. 256 loop
            Set_Flat (Q, QP, Bk (I));
            QP := QP + 1;
         end loop;
      end loop;
   end Dequant_Q6_K;

   --  Q3_K super-block (llama.cpp block_q3_K, 110 bytes / 256 elements):
   --    uint8_t hmask[32]; -- 3rd (high) bit of each 3-bit quant
   --    uint8_t qs[64];    -- low 2 bits, 4 quants per byte
   --    uint8_t scales[12];-- 16 signed 6-bit scales, packed (get from kmask1/2)
   --    ggml_half d;       -- super-block scale
   --  Value: y = d*(scale[is]-32)*((qs_2bit | hbit<<2) - 4), per 16-element group.
   --  Reference: llama.cpp dequantize_row_q3_K (m shifts per j across 2 halves).
   procedure Decode_Q3K_Block (X : String; Pos : Natural; B : out Block256) is
      function U8 (P : Natural) return Unsigned_32 is
        (Unsigned_32 (Character'Pos (X (P))));
      function LE32 (P : Natural) return Unsigned_32 is
        (U8 (P) or Shift_Left (U8 (P + 1), 8)
         or Shift_Left (U8 (P + 2), 16) or Shift_Left (U8 (P + 3), 24));
      HM_Base : constant Natural := Pos;          -- hmask[32]
      QS_Base : constant Natural := Pos + 32;     -- qs[64]
      SC_Base : constant Natural := Pos + 96;     -- scales[12]
      D_All   : constant Float := F16_To_F32
        (Byte (Character'Pos (X (Pos + 108))),
         Byte (Character'Pos (X (Pos + 109))));
      KM1 : constant Unsigned_32 := 16#0303_0303#;   -- low 2 bits per byte
      KM2 : constant Unsigned_32 := 16#0F0F_0F0F#;   -- low 4 bits per byte
      A0  : constant Unsigned_32 := LE32 (SC_Base);
      A1  : constant Unsigned_32 := LE32 (SC_Base + 4);
      Tmp : constant Unsigned_32 := LE32 (SC_Base + 8);
      --  16 six-bit scales spread across these four words (4 bytes each):
      S0  : constant Unsigned_32 :=
        (A0 and KM2) or Shift_Left (Shift_Right (Tmp, 0) and KM1, 4);
      S1  : constant Unsigned_32 :=
        (A1 and KM2) or Shift_Left (Shift_Right (Tmp, 2) and KM1, 4);
      S2  : constant Unsigned_32 :=
        (Shift_Right (A0, 4) and KM2) or Shift_Left (Shift_Right (Tmp, 4) and KM1, 4);
      S3  : constant Unsigned_32 :=
        (Shift_Right (A1, 4) and KM2) or Shift_Left (Shift_Right (Tmp, 6) and KM1, 4);

      function Scale (Idx : Natural) return Integer is
         W : Unsigned_32;
         K : constant Natural := Idx mod 4;
      begin
         case Idx / 4 is
            when 0      => W := S0;
            when 1      => W := S1;
            when 2      => W := S2;
            when others => W := S3;
         end case;
         return Integer (Shift_Right (W, 8 * K) and 16#FF#) - 32;
      end Scale;
   begin
      for P in 0 .. 255 loop
         declare
            NH   : constant Natural := P / 128;        -- which 128-half
            Pp   : constant Natural := P mod 128;
            J    : constant Natural := Pp / 32;        -- 2-bit field (shift 2*J)
            Rm   : constant Natural := Pp mod 32;
            G    : constant Natural := Rm / 16;        -- group {0,1} within the 32
            L    : constant Natural := Rm mod 16;
            QIdx : constant Natural := QS_Base + NH * 32 + G * 16 + L;
            HIdx : constant Natural := HM_Base + G * 16 + L;
            Low2 : constant Integer :=
              Integer (Shift_Right (U8 (QIdx), 2 * J) and 3);
            HBit : constant Integer :=
              (if (U8 (HIdx) and Shift_Left (Unsigned_32 (1), NH * 4 + J)) /= 0
               then 1 else 0);
            Sg   : constant Integer := Low2 + 4 * HBit - 4;     -- signed -4..3
            Iss  : constant Natural := NH * 8 + J * 2 + G;
         begin
            B (P + 1) := D_All * Float (Scale (Iss)) * Float (Sg);
         end;
      end loop;
   end Decode_Q3K_Block;

   procedure Dequant_Q3_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Bk     : Block256;
      QP     : Natural := 1;
   begin
      for Blk in 0 .. Blocks - 1 loop
         Decode_Q3K_Block (X, X'First + Blk * 110, Bk);
         for I in 1 .. 256 loop
            Set_Flat (Q, QP, Bk (I));
            QP := QP + 1;
         end loop;
      end loop;
   end Dequant_Q3_K;

   --  Q2_K super-block (llama.cpp block_q2_K, 84 bytes / 256 elements):
   --    uint8_t scales[16]; -- per-16: 4-bit scale (low nibble) + 4-bit min (high)
   --    uint8_t qs[64];     -- 2-bit quants, 4 per byte
   --    ggml_half d;        -- super-block scale for the 4-bit scales
   --    ggml_half dmin;     -- super-block scale for the 4-bit mins
   --  Value: y = d*(sc & 0xF)*q2 - dmin*(sc >> 4), per 16-element group.
   --  Reference: llama.cpp dequantize_row_q2_K.
   procedure Decode_Q2K_Block (X : String; Pos : Natural; B : out Block256) is
      function U8 (P : Natural) return Unsigned_32 is
        (Unsigned_32 (Character'Pos (X (P))));
      SC_Base : constant Natural := Pos;          -- scales[16]
      QS_Base : constant Natural := Pos + 16;     -- qs[64]
      D       : constant Float := F16_To_F32
        (Byte (Character'Pos (X (Pos + 80))),
         Byte (Character'Pos (X (Pos + 81))));
      DMin    : constant Float := F16_To_F32
        (Byte (Character'Pos (X (Pos + 82))),
         Byte (Character'Pos (X (Pos + 83))));
   begin
      for P in 0 .. 255 loop
         declare
            NH   : constant Natural := P / 128;
            Pp   : constant Natural := P mod 128;
            J    : constant Natural := Pp / 32;        -- 2-bit field (shift 2*J)
            Rm   : constant Natural := Pp mod 32;
            G    : constant Natural := Rm / 16;        -- group {0,1}
            L    : constant Natural := Rm mod 16;
            QIdx : constant Natural := QS_Base + NH * 32 + G * 16 + L;
            Iss  : constant Natural := NH * 8 + J * 2 + G;
            Sc   : constant Unsigned_32 := U8 (SC_Base + Iss);
            Q2   : constant Float :=
              Float (Shift_Right (U8 (QIdx), 2 * J) and 3);
            Dl   : constant Float := D    * Float (Sc and 16#0F#);
            Ml   : constant Float := DMin * Float (Shift_Right (Sc, 4));
         begin
            B (P + 1) := Dl * Q2 - Ml;
         end;
      end loop;
   end Decode_Q2K_Block;

   procedure Dequant_Q2_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Bk     : Block256;
      QP     : Natural := 1;
   begin
      for Blk in 0 .. Blocks - 1 loop
         Decode_Q2K_Block (X, X'First + Blk * 84, Bk);
         for I in 1 .. 256 loop
            Set_Flat (Q, QP, Bk (I));
            QP := QP + 1;
         end loop;
      end loop;
   end Dequant_Q2_K;

   --------------------------------------------------------------------
   -- Q4_K dequantization
   --
   -- Super-block of 256 elements (llama.cpp block_q4_K, 144 bytes):
   --   - 2 bytes:  FP16 d    (super-block scale for the 6-bit scales)
   --   - 2 bytes:  FP16 dmin (super-block scale for the 6-bit mins)
   --   - 12 bytes: scales[12] — eight 6-bit scales + eight 6-bit mins, packed
   --   - 128 bytes: qs (256 × 4-bit quants, low then high nibble of each byte)
   --
   -- 256 elements are emitted in 4 groups of 64: within each group the low
   -- nibbles of qs[0..31] use (d1,m1), the high nibbles use (d2,m2), where
   -- (sc,m) come from get_scale_min_k4.  Value = d*sc*q - dmin*m  (min subtracted).
   --
   -- Reference: llama.cpp, dequantize_row_q4_K / get_scale_min_k4.
   --------------------------------------------------------------------

   procedure Dequant_Q4_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;

      function U8 (P : Natural) return Byte is (Byte (Character'Pos (X (P))));
   begin
      for B in 1 .. Blocks loop
         declare
            D    : constant Float := F16_To_F32 (U8 (Pos),     U8 (Pos + 1));
            DMin : constant Float := F16_To_F32 (U8 (Pos + 2), U8 (Pos + 3));
            Sc   : array (0 .. 11) of Byte;          -- the 12 packed scale bytes
            QS_Base : constant Natural := Pos + 16;  -- start of qs (128 bytes)

            --  Extract the J-th (0..7) 6-bit scale (Dd) and min (Mm).
            procedure Get_SM (J : Natural; Dd, Mm : out Byte) is
            begin
               if J < 4 then
                  Dd := Sc (J)     and 63;
                  Mm := Sc (J + 4) and 63;
               else
                  Dd := (Sc (J + 4) and 16#0F#)
                          or Shift_Left (Shift_Right (Sc (J - 4), 6), 4);
                  Mm := Shift_Right (Sc (J + 4), 4)
                          or Shift_Left (Shift_Right (Sc (J), 6), 4);
               end if;
            end Get_SM;

            Is_Idx : Natural := 0;
         begin
            for I in 0 .. 11 loop Sc (I) := U8 (Pos + 4 + I); end loop;

            for G in 0 .. 3 loop                     -- the 4 groups of 64 elems
               declare
                  Sc1, M1b, Sc2, M2b : Byte;
                  QOff : constant Natural := QS_Base + G * 32;
               begin
                  Get_SM (Is_Idx,     Sc1, M1b);
                  Get_SM (Is_Idx + 1, Sc2, M2b);
                  declare
                     D1 : constant Float := D * Float (Integer (Sc1));
                     M1 : constant Float := DMin * Float (Integer (M1b));
                     D2 : constant Float := D * Float (Integer (Sc2));
                     M2 : constant Float := DMin * Float (Integer (M2b));
                  begin
                     for L in 0 .. 31 loop          -- low nibbles
                        Set_Flat (Q, Q_Pos,
                          D1 * Float (Integer (U8 (QOff + L) and 16#0F#)) - M1);
                        Q_Pos := Q_Pos + 1;
                     end loop;
                     for L in 0 .. 31 loop          -- high nibbles
                        Set_Flat (Q, Q_Pos,
                          D2 * Float (Integer (Shift_Right (U8 (QOff + L), 4))) - M2);
                        Q_Pos := Q_Pos + 1;
                     end loop;
                  end;
                  Is_Idx := Is_Idx + 2;
               end;
            end loop;
            Pos := Pos + 144;
         end;
      end loop;
   end Dequant_Q4_K;

   --  Q5_K super-block (llama.cpp block_q5_K, 176 bytes / 256 elements):
   --    ggml_half d;        -- f16 super-block scale for the 6-bit scales
   --    ggml_half dmin;     -- f16 super-block scale for the 6-bit mins
   --    uint8_t scales[12]; -- 8 sub-block scales + 8 mins, 6-bit packed
   --    uint8_t qh[32];     -- 5th (high) bit of each quant
   --    uint8_t qs[128];    -- low 4 bits, 2 quants per byte
   --  Value: y = d*sc*((qs&0xF | qh_bit<<4)) - dmin*m, per 32-element sub-block.
   --  Decode ONE Q5_K super-block (176 bytes at Pos) into B (1..256).
   procedure Decode_Q5K_Block (X : String; Pos : Natural; B : out Block256) is
      function RF16 (P : Natural) return Float is
        (F16_To_F32 (Byte (Character'Pos (X (P))),
                     Byte (Character'Pos (X (P + 1)))));
      D    : constant Float := RF16 (Pos);
      DMin : constant Float := RF16 (Pos + 2);
      Sc   : array (0 .. 11) of Unsigned_32;
      QH   : array (0 .. 31) of Unsigned_32;
      QS   : array (0 .. 127) of Unsigned_32;

      procedure Scale_Min (J : Natural; Scl, Mn : out Unsigned_32) is
      begin
         if J < 4 then
            Scl := Sc (J) and 63;
            Mn  := Sc (J + 4) and 63;
         else
            Scl := (Sc (J + 4) and 16#0F#)
                   or Shift_Left (Shift_Right (Sc (J - 4), 6), 4);
            Mn  := Shift_Right (Sc (J + 4), 4)
                   or Shift_Left (Shift_Right (Sc (J), 6), 4);
         end if;
      end Scale_Min;

      P      : Natural := Pos + 4;
   begin
      for I in 0 .. 11 loop
         Sc (I) := Unsigned_32 (Character'Pos (X (P + I)));
      end loop;
      P := P + 12;
      for I in 0 .. 31 loop
         QH (I) := Unsigned_32 (Character'Pos (X (P + I)));
      end loop;
      P := P + 32;
      for I in 0 .. 127 loop
         QS (I) := Unsigned_32 (Character'Pos (X (P + I)));
      end loop;

      for G in 0 .. 3 loop
         declare
            Sc1, M1, Sc2, M2 : Unsigned_32;
            Sh1 : constant Natural := 2 * G;       -- 5th-bit position, low nibble
            Sh2 : constant Natural := 2 * G + 1;   -- 5th-bit position, high nibble
            QB  : constant Natural := 32 * G;      -- qs base for this group
            B1  : constant Natural := 64 * G;      -- 0-based output base, low half
            B2  : constant Natural := 64 * G + 32; -- 0-based output base, high half
         begin
            Scale_Min (2 * G,     Sc1, M1);
            Scale_Min (2 * G + 1, Sc2, M2);
            declare
               D1  : constant Float := D * Float (Sc1);
               Mn1 : constant Float := DMin * Float (M1);
               D2  : constant Float := D * Float (Sc2);
               Mn2 : constant Float := DMin * Float (M2);
            begin
               --  Branchless 5th bit (shift, no per-element branch) and fixed
               --  index bases (no loop-carried counter) so the int->float +
               --  scale of both nibble halves vectorises to NEON.
               for L in 0 .. 31 loop
                  declare
                     Q   : constant Unsigned_32 := QS (QB + L);
                     Lo  : constant Unsigned_32 := (Q and 16#0F#)
                       + Shift_Left (Shift_Right (QH (L), Sh1) and 1, 4);
                     Hi  : constant Unsigned_32 := Shift_Right (Q, 4)
                       + Shift_Left (Shift_Right (QH (L), Sh2) and 1, 4);
                  begin
                     B (B1 + L + 1) := D1 * Float (Lo) - Mn1;
                     B (B2 + L + 1) := D2 * Float (Hi) - Mn2;
                  end;
               end loop;
            end;
         end;
      end loop;
   end Decode_Q5K_Block;

   --  Decode one 256-element Q4_K super-block (144 bytes) into B (1 .. 256),
   --  in the SAME element order Dequant_Q4_K produces. Logic mirrors the
   --  (independently validated) Dequant_Q4_K bulk path; keeping the bulk
   --  routine untouched lets the QMatVec test cross-check this against it.
   procedure Decode_Q4K_Block (X : String; Pos : Natural; B : out Block256) is
      function U8 (P : Natural) return Byte is (Byte (Character'Pos (X (P))));
      D    : constant Float := F16_To_F32 (U8 (Pos),     U8 (Pos + 1));
      DMin : constant Float := F16_To_F32 (U8 (Pos + 2), U8 (Pos + 3));
      Sc   : array (0 .. 11) of Byte;          -- the 12 packed scale bytes
      QS_Base : constant Natural := Pos + 16;  -- start of qs (128 bytes)
      Idx    : Natural := 1;                   -- 1-based write cursor into B
      Is_Idx : Natural := 0;

      --  Extract the J-th (0..7) 6-bit scale (Dd) and min (Mm).
      procedure Get_SM (J : Natural; Dd, Mm : out Byte) is
      begin
         if J < 4 then
            Dd := Sc (J)     and 63;
            Mm := Sc (J + 4) and 63;
         else
            Dd := (Sc (J + 4) and 16#0F#)
                    or Shift_Left (Shift_Right (Sc (J - 4), 6), 4);
            Mm := Shift_Right (Sc (J + 4), 4)
                    or Shift_Left (Shift_Right (Sc (J), 6), 4);
         end if;
      end Get_SM;
   begin
      for I in 0 .. 11 loop Sc (I) := U8 (Pos + 4 + I); end loop;

      for G in 0 .. 3 loop                     -- the 4 groups of 64 elems
         declare
            Sc1, M1b, Sc2, M2b : Byte;
            QOff : constant Natural := QS_Base + G * 32;
         begin
            Get_SM (Is_Idx,     Sc1, M1b);
            Get_SM (Is_Idx + 1, Sc2, M2b);
            declare
               D1 : constant Float := D * Float (Integer (Sc1));
               M1 : constant Float := DMin * Float (Integer (M1b));
               D2 : constant Float := D * Float (Integer (Sc2));
               M2 : constant Float := DMin * Float (Integer (M2b));
            begin
               for L in 0 .. 31 loop          -- low nibbles
                  B (Idx) := D1 * Float (Integer (U8 (QOff + L) and 16#0F#)) - M1;
                  Idx := Idx + 1;
               end loop;
               for L in 0 .. 31 loop          -- high nibbles
                  B (Idx) := D2 * Float (Integer (Shift_Right (U8 (QOff + L), 4))) - M2;
                  Idx := Idx + 1;
               end loop;
            end;
            Is_Idx := Is_Idx + 2;
         end;
      end loop;
   end Decode_Q4K_Block;

   procedure Dequant_Q5_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Bk     : Block256;
      QP     : Natural := 1;
   begin
      for Blk in 0 .. Blocks - 1 loop
         Decode_Q5K_Block (X, X'First + Blk * 176, Bk);
         for I in 1 .. 256 loop
            Set_Flat (Q, QP, Bk (I));
            QP := QP + 1;
         end loop;
      end loop;
   end Dequant_Q5_K;

   --------------------------------------------------------------------
   -- Generic dequantize dispatcher
   --------------------------------------------------------------------

   function Dequant_Num_Elements (Info : LLM_GGUF.Tensor_Info) return Natural is
      N : Long_Long_Integer := 1;
   begin
      for D in 1 .. Natural (Info.N_Dims) loop
         N := N * Long_Long_Integer (Info.Dims (D));
      end loop;
      return Natural (N);
   end Dequant_Num_Elements;

   --  Row-major logical shape: GGUF lists dims fastest-varying first, so the
   --  contiguous (row-major) shape is the GGUF dims reversed. E.g. token_embd
   --  GGUF [dim, vocab] -> logical [vocab, dim], so Get([token, d]) is correct.
   function Logical_Shape (Info : LLM_GGUF.Tensor_Info) return Dims is
      ND : constant Natural := Natural (Info.N_Dims);
   begin
      if ND = 0 then
         return [1 => 1];
      end if;
      return S : Dims (1 .. ND) do
         for I in 1 .. ND loop
            S (I) := Positive (Info.Dims (ND - I + 1));
         end loop;
      end return;
   end Logical_Shape;

   --  Q8_0: 32-element blocks of [f16 scale d | 32 x int8 q]; w = d * q.
   procedure Dequant_Q8_0 (X : String; Q : out Tensor; N : Natural) is
      Pos : Natural := X'First;
      Idx : Natural := 0;
   begin
      while Idx < N loop
         declare
            D : constant Float := F16_To_F32
              (Byte (Character'Pos (X (Pos))),
               Byte (Character'Pos (X (Pos + 1))));
         begin
            Pos := Pos + 2;
            for J in 0 .. 31 loop
               exit when Idx >= N;
               declare
                  B : constant Integer := Character'Pos (X (Pos + J));
                  S : constant Integer := (if B >= 128 then B - 256 else B);
               begin
                  Set_Flat (Q, Idx + 1, D * Float (S));
               end;
               Idx := Idx + 1;
            end loop;
            Pos := Pos + 32;
         end;
      end loop;
   end Dequant_Q8_0;

   --------------------------------------------------------------------
   -- Q4_0: 32-element block = f16 d (2) + qs (16). Two nibbles per byte map
   -- to positions j and j+16; value = (nibble - 8) * d. (llama.cpp block_q4_0.)
   --------------------------------------------------------------------
   procedure Dequant_Q4_0 (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 32;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;
      function U8 (P : Natural) return Byte is (Byte (Character'Pos (X (P))));
   begin
      for B in 1 .. Blocks loop
         declare
            D  : constant Float := F16_To_F32 (U8 (Pos), U8 (Pos + 1));
            QB : constant Natural := Pos + 2;
         begin
            for J in 0 .. 15 loop
               declare
                  C  : constant Byte := U8 (QB + J);
                  X0 : constant Integer := Integer (C and 16#0F#) - 8;
                  X1 : constant Integer := Integer (Shift_Right (C, 4)) - 8;
               begin
                  Set_Flat (Q, Q_Pos + J,      D * Float (X0));
                  Set_Flat (Q, Q_Pos + J + 16, D * Float (X1));
               end;
            end loop;
            Q_Pos := Q_Pos + 32;
            Pos   := Pos + 18;
         end;
      end loop;
   end Dequant_Q4_0;

   --------------------------------------------------------------------
   -- Q5_0: 32-element block = f16 d (2) + qh (4, the 5th bit per element as a
   -- LE uint32) + qs (16). value = ((low4 | high5th) - 16) * d.
   --------------------------------------------------------------------
   procedure Dequant_Q5_0 (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 32;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;
      function U8 (P : Natural) return Byte is (Byte (Character'Pos (X (P))));
   begin
      for B in 1 .. Blocks loop
         declare
            D  : constant Float := F16_To_F32 (U8 (Pos), U8 (Pos + 1));
            QH : constant Unsigned_32 :=
                 Unsigned_32 (U8 (Pos + 2))
               + Shift_Left (Unsigned_32 (U8 (Pos + 3)), 8)
               + Shift_Left (Unsigned_32 (U8 (Pos + 4)), 16)
               + Shift_Left (Unsigned_32 (U8 (Pos + 5)), 24);
            QB : constant Natural := Pos + 6;
         begin
            for J in 0 .. 15 loop
               declare
                  C   : constant Byte := U8 (QB + J);
                  XH0 : constant Integer :=
                    Integer (Shift_Left (Shift_Right (QH, J), 4) and 16#10#);
                  XH1 : constant Integer :=
                    Integer (Shift_Right (QH, J + 12) and 16#10#);
                  X0  : constant Integer := (Integer (C and 16#0F#) + XH0) - 16;
                  X1  : constant Integer := (Integer (Shift_Right (C, 4)) + XH1) - 16;
               begin
                  Set_Flat (Q, Q_Pos + J,      D * Float (X0));
                  Set_Flat (Q, Q_Pos + J + 16, D * Float (X1));
               end;
            end loop;
            Q_Pos := Q_Pos + 32;
            Pos   := Pos + 22;
         end;
      end loop;
   end Dequant_Q5_0;

   --  Dequantize into a caller-provided (already-allocated) tensor — no heap
   --  allocation, so it can be called in a hot/parallel loop with a reused buffer.
   procedure Fill
     (Info : LLM_GGUF.Tensor_Info; Raw : String; Result : in out Tensor; N : Natural)
   is
   begin
      case Info.Kind is
         when LLM_GGUF.GGML_TYPE_F32 =>
            declare
               Pos : Natural := Raw'First;
            begin
               for I in 1 .. N loop
                  declare
                     subtype Float32 is String (1 .. 4);
                     F_Bytes : Float32;
                     F : Float;
                     for F'Address use F_Bytes'Address;
                  begin
                     F_Bytes := Raw (Pos .. Pos + 3);
                     Pos := Pos + 4;
                     Set_Flat (Result, I, F);
                  end;
               end loop;
            end;

         when LLM_GGUF.GGML_TYPE_F16 =>
            for I in 1 .. N loop
               declare
                  Pos : constant Natural := Raw'First + (I - 1) * 2;
                  F : constant Float := F16_To_F32 (
                    Byte (Character'Pos (Raw (Pos))),
                    Byte (Character'Pos (Raw (Pos + 1))));
               begin
                  Set_Flat (Result, I, F);
               end;
            end loop;

         when LLM_GGUF.GGML_TYPE_BF16 =>
            --  BF16 is the high 16 bits of an F32: place the two BF16 bytes in
            --  the high half of a 4-byte F32 (low half zero), reinterpret.
            for I in 1 .. N loop
               declare
                  Pos : constant Natural := Raw'First + (I - 1) * 2;
                  Bytes : String (1 .. 4);
                  F     : Float;
                  for F'Address use Bytes'Address;
               begin
                  Bytes (1) := Character'Val (0);
                  Bytes (2) := Character'Val (0);
                  Bytes (3) := Raw (Pos);
                  Bytes (4) := Raw (Pos + 1);
                  Set_Flat (Result, I, F);
               end;
            end loop;

         when LLM_GGUF.GGML_TYPE_Q5_K => Dequant_Q5_K (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q8_K => Dequant_Q8_K (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q6_K => Dequant_Q6_K (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q4_K => Dequant_Q4_K (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q3_K => Dequant_Q3_K (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q2_K => Dequant_Q2_K (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q8_0 => Dequant_Q8_0 (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q4_0 => Dequant_Q4_0 (Raw, Result, N);
         when LLM_GGUF.GGML_TYPE_Q5_0 => Dequant_Q5_0 (Raw, Result, N);

         when others =>
            --  Do NOT silently zero-fill: that yields a model that loads and
            --  then emits garbage. Fail loudly so the caller (model load) can
            --  reject the file with a clear, actionable message.
            raise Unsupported_Quant with
              "unsupported GGML quantization type "
              & LLM_GGUF.GGML_Type'Image (Info.Kind)
              & " (e.g. IQ* / ternary types are not implemented)";
      end case;
   end Fill;

   function Is_Supported (Kind : LLM_GGUF.GGML_Type) return Boolean is
      use type LLM_GGUF.GGML_Type;
   begin
      return Kind in
        LLM_GGUF.GGML_TYPE_F32 | LLM_GGUF.GGML_TYPE_F16
        | LLM_GGUF.GGML_TYPE_BF16
        | LLM_GGUF.GGML_TYPE_Q8_0 | LLM_GGUF.GGML_TYPE_Q4_0
        | LLM_GGUF.GGML_TYPE_Q5_0
        | LLM_GGUF.GGML_TYPE_Q2_K | LLM_GGUF.GGML_TYPE_Q3_K
        | LLM_GGUF.GGML_TYPE_Q4_K | LLM_GGUF.GGML_TYPE_Q5_K
        | LLM_GGUF.GGML_TYPE_Q6_K | LLM_GGUF.GGML_TYPE_Q8_K;
   end Is_Supported;

   function Dequantize
     (Info : LLM_GGUF.Tensor_Info;
      Raw  : String)
      return Tensor
   is
      N : constant Natural := Dequant_Num_Elements (Info);
   begin
      return Result : Tensor := New_Tensor (Logical_Shape (Info)) do
         Fill (Info, Raw, Result, N);
      end return;
   end Dequantize;

   --------------------------------------------------------------------
   -- Streaming quantized matrix-vector
   --------------------------------------------------------------------

   function QMatVec
     (Info : LLM_GGUF.Tensor_Info;
      Raw  : String;
      X    : Tensor)
      return Tensor
   is
      use type LLM_GGUF.GGML_Type;
      In_Dim   : constant Integer := Integer (Info.Dims (1));   -- GGUF ne0 = in
      Out_Dim  : constant Integer := Integer (Info.Dims (2));   -- GGUF ne1 = out
      Kind     : constant LLM_GGUF.GGML_Type := Info.Kind;
      Is_Q5K   : constant Boolean := Kind = LLM_GGUF.GGML_TYPE_Q5_K;
      Is_Q6K   : constant Boolean := Kind = LLM_GGUF.GGML_TYPE_Q6_K;
      Is_Q4K   : constant Boolean := Kind = LLM_GGUF.GGML_TYPE_Q4_K;
      Is_Q3K   : constant Boolean := Kind = LLM_GGUF.GGML_TYPE_Q3_K;
      Is_Q2K   : constant Boolean := Kind = LLM_GGUF.GGML_TYPE_Q2_K;
      Fused    : constant Boolean :=
        Is_Q5K or else Is_Q6K or else Is_Q4K or else Is_Q3K or else Is_Q2K;
      N_Blk    : constant Integer := In_Dim / 256;
      Blk_Bytes : constant Integer :=
        (if Is_Q5K then 176 elsif Is_Q6K then 210 elsif Is_Q4K then 144
         elsif Is_Q3K then 110 elsif Is_Q2K then 84 else 0);
      Row_Info : LLM_GGUF.Tensor_Info := Info;
      BPR      : Natural;

      --  Local FP32 copy of x — the hot dot loop reads it as a plain array
      --  (no Tensor access indirection), so the compiler can vectorise (FMA).
      XL : array (1 .. In_Dim) of Float;
   begin
      --  K-quants only exist for 256-element super-blocks; the fused path
      --  drops a partial trailing block. Guard explicitly (checks are
      --  suppressed in this unit) so a malformed tensor fails loudly here
      --  rather than silently producing a wrong dot product.
      if Fused and then In_Dim mod 256 /= 0 then
         raise Constraint_Error
           with "QMatVec: K-quant in-dim" & Integer'Image (In_Dim)
                & " is not a multiple of 256";
      end if;

      --  The input vector must match the weight's in-dimension. Checks are
      --  suppressed in this unit, so verify explicitly: otherwise a mismatched
      --  X would read out of bounds (silent garbage) in the XL fill below.
      if Numel (X) /= In_Dim then
         raise Constraint_Error
           with "QMatVec: input length" & Integer'Image (Numel (X))
                & " /= weight in-dim" & Integer'Image (In_Dim);
      end if;

      Row_Info.N_Dims := 2;
      Row_Info.Dims   := [Info.Dims (1), 1, 0, 0];
      BPR := Natural (LLM_GGUF.Tensor_Byte_Size (Row_Info));
      for I in 1 .. In_Dim loop
         XL (I) := Get_Flat (X, I);
      end loop;

      return Y : Tensor := New_Tensor ([1, Out_Dim]) do
         declare
            type Rows_Op is new LLM_Pool.Parallel_Op with null record;
            overriding procedure Execute (Op : in out Rows_Op; Lo, Hi : Integer) is
               Bk    : Block256;          -- one decoded super-block (stack)
               Row_T : Tensor;            -- fallback buffer (non-K-quant only)
            begin
               if Fused then
                  for O in Lo .. Hi loop
                     declare
                        RS  : constant Natural := Raw'First + (O - 1) * BPR;
                        --  Four independent accumulator lanes break the FP
                        --  reduction dependency chain so the compiler packs the
                        --  fused decode+dot into NEON vector FMAs. (Sum order
                        --  differs from a scalar reduction by ~1e-3 rounding.)
                        A0, A1, A2, A3 : Float := 0.0;
                     begin
                        for Blk in 0 .. N_Blk - 1 loop
                           if Is_Q5K then
                              Decode_Q5K_Block (Raw, RS + Blk * Blk_Bytes, Bk);
                           elsif Is_Q4K then
                              Decode_Q4K_Block (Raw, RS + Blk * Blk_Bytes, Bk);
                           elsif Is_Q6K then
                              Decode_Q6K_Block (Raw, RS + Blk * Blk_Bytes, Bk);
                           elsif Is_Q3K then
                              Decode_Q3K_Block (Raw, RS + Blk * Blk_Bytes, Bk);
                           else
                              Decode_Q2K_Block (Raw, RS + Blk * Blk_Bytes, Bk);
                           end if;
                           declare
                              Base : constant Integer := Blk * 256;
                           begin
                              for K in 0 .. 63 loop
                                 A0 := A0 + Bk (4 * K + 1) * XL (Base + 4 * K + 1);
                                 A1 := A1 + Bk (4 * K + 2) * XL (Base + 4 * K + 2);
                                 A2 := A2 + Bk (4 * K + 3) * XL (Base + 4 * K + 3);
                                 A3 := A3 + Bk (4 * K + 4) * XL (Base + 4 * K + 4);
                              end loop;
                           end;
                        end loop;
                        Set_Flat (Y, O, (A0 + A1) + (A2 + A3));
                     end;
                  end loop;
               else
                  --  Generic fallback (F32/F16/…): dequant row, then dot.
                  Row_T := New_Tensor ([1, In_Dim]);
                  for O in Lo .. Hi loop
                     declare
                        RS  : constant Natural := Raw'First + (O - 1) * BPR;
                        Acc : Float := 0.0;
                     begin
                        Fill (Row_Info, Raw (RS .. RS + BPR - 1), Row_T, In_Dim);
                        for I in 1 .. In_Dim loop
                           Acc := Acc + Get_Flat (Row_T, I) * XL (I);
                        end loop;
                        Set_Flat (Y, O, Acc);
                     end;
                  end loop;
               end if;
            end Execute;

            Op : Rows_Op;
         begin
            LLM_Pool.Run (Op, 1, Out_Dim, Min_Grain => 2048);
         end;
      end return;
   end QMatVec;

end LLM_Dequant;
