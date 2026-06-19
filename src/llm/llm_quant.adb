---------------------------------------------------------------------
-- LLM_Quant body.
---------------------------------------------------------------------

with Ada.Unchecked_Conversion;
with Interfaces; use Interfaces;

package body LLM_Quant is

   procedure F32_To_F16 (X : Float; Lo, Hi : out Character) is
      function To_U32 is new Ada.Unchecked_Conversion (Float, Unsigned_32);
      B    : constant Unsigned_32 := To_U32 (X);
      Sign : constant Unsigned_16 := Unsigned_16 (Shift_Right (B, 16) and 16#8000#);
      Exp  : constant Integer     := Integer (Shift_Right (B, 23) and 16#FF#);
      Mant : constant Unsigned_32 := B and 16#7F_FFFF#;
      H    : Unsigned_16;
      E    : Integer;
   begin
      if Exp = 16#FF# then                      -- inf / nan
         H := Sign or 16#7C00# or (if Mant /= 0 then 16#0200# else 0);
      else
         E := Exp - 112;                         -- (Exp - 127) + 15
         if E >= 16#1F# then
            H := Sign or 16#7C00#;               -- overflow -> inf
         elsif E <= 0 then
            if E < -10 then
               H := Sign;                         -- underflow -> zero
            else
               declare
                  M  : constant Unsigned_32 := Mant or 16#80_0000#;
                  Sh : constant Natural     := Natural (14 - E);
               begin
                  H := Sign or Unsigned_16 (Shift_Right (M, Sh) and 16#03FF#);
               end;
            end if;
         else                                     -- normal, round to nearest
            declare
               Frac  : constant Unsigned_16 :=
                 Unsigned_16 (Shift_Right (Mant, 13) and 16#03FF#);
               Round : constant Boolean :=
                 (Mant and 16#1000#) /= 0
                 and then ((Mant and 16#0FFF#) /= 0
                           or else (Shift_Right (Mant, 13) and 1) /= 0);
            begin
               H := Sign or Unsigned_16 (E * 1024) or Frac;
               if Round then H := H + 1; end if;   -- may carry into exponent
            end;
         end if;
      end if;
      Lo := Character'Val (Integer (H and 16#FF#));
      Hi := Character'Val (Integer (Shift_Right (H, 8) and 16#FF#));
   end F32_To_F16;

   function Quantize_Q8_0 (X : Tensor) return String is
      N      : constant Natural := Numel (X);
      Blocks : constant Natural := N / 32;
      R      : String (1 .. Blocks * 34);
      P      : Natural := 1;
   begin
      for Blk in 0 .. Blocks - 1 loop
         declare
            Base : constant Natural := Blk * 32;
            Amax : Float := 0.0;
            D, Inv : Float;
            Lo, Hi : Character;
         begin
            for J in 0 .. 31 loop
               declare V : constant Float := abs (Get_Flat (X, Base + J + 1));
               begin if V > Amax then Amax := V; end if; end;
            end loop;
            D := Amax / 127.0;
            F32_To_F16 (D, Lo, Hi);            -- store the f16 scale
            R (P) := Lo; R (P + 1) := Hi; P := P + 2;

            Inv := (if D > 0.0 then 1.0 / D else 0.0);   -- ggml: quantize with f32 1/d
            for J in 0 .. 31 loop
               declare
                  Q : Integer :=
                    Integer (Float'Rounding (Get_Flat (X, Base + J + 1) * Inv));
                  U : Integer;
               begin
                  if Q > 127 then Q := 127; elsif Q < -127 then Q := -127; end if;
                  U := (if Q < 0 then Q + 256 else Q);    -- signed -> unsigned byte
                  R (P) := Character'Val (U);
                  P := P + 1;
               end;
            end loop;
         end;
      end loop;
      return R;
   end Quantize_Q8_0;

   function Quantize_Q4_0 (X : Tensor) return String is
      N      : constant Natural := Numel (X);
      Blocks : constant Natural := N / 32;
      R      : String (1 .. Blocks * 18);
      P      : Natural := 1;
   begin
      for Blk in 0 .. Blocks - 1 loop
         declare
            Base   : constant Natural := Blk * 32;
            Amax   : Float := 0.0;
            Vmax   : Float := 0.0;        -- signed value at the max-abs position
            D, Id  : Float;
            Lo, Hi : Character;
            Q      : array (0 .. 31) of Integer;
         begin
            for J in 0 .. 31 loop
               declare V : constant Float := Get_Flat (X, Base + J + 1);
               begin if abs V > Amax then Amax := abs V; Vmax := V; end if; end;
            end loop;
            --  ggml convention: d = vmax / -8, so the max-abs element is exact.
            D := (if Amax > 0.0 then Vmax / (-8.0) else 0.0);
            F32_To_F16 (D, Lo, Hi);
            R (P) := Lo; R (P + 1) := Hi; P := P + 2;

            Id := (if D /= 0.0 then 1.0 / D else 0.0);
            for J in 0 .. 31 loop
               declare
                  QV : Integer :=
                    Integer (Float'Rounding (Get_Flat (X, Base + J + 1) * Id)) + 8;
               begin
                  if QV < 0 then QV := 0; elsif QV > 15 then QV := 15; end if;
                  Q (J) := QV;
               end;
            end loop;
            --  pack: byte J holds q[J] (low nibble) + q[J+16] (high nibble).
            for J in 0 .. 15 loop
               R (P) := Character'Val (Q (J) + 16 * Q (J + 16));
               P := P + 1;
            end loop;
         end;
      end loop;
      return R;
   end Quantize_Q4_0;

   function Quantize_Q5_0 (X : Tensor) return String is
      N      : constant Natural := Numel (X);
      Blocks : constant Natural := N / 32;
      R      : String (1 .. Blocks * 22);
      P      : Natural := 1;
      procedure Put_Byte (Bt : Natural) is
      begin R (P) := Character'Val (Bt mod 256); P := P + 1; end Put_Byte;
   begin
      for Blk in 0 .. Blocks - 1 loop
         declare
            Base   : constant Natural := Blk * 32;
            Amax   : Float := 0.0;
            Vmax   : Float := 0.0;        -- signed value at the max-abs position
            D, Id  : Float;
            Lo, Hi : Character;
            Q      : array (0 .. 31) of Integer;
            QH     : Unsigned_32 := 0;    -- 32 high bits (uint32, little-endian)
         begin
            for J in 0 .. 31 loop
               declare V : constant Float := Get_Flat (X, Base + J + 1);
               begin if abs V > Amax then Amax := abs V; Vmax := V; end if; end;
            end loop;
            --  ggml convention: d = vmax / -16, so the max-abs element is exact.
            D := (if Amax > 0.0 then Vmax / (-16.0) else 0.0);
            F32_To_F16 (D, Lo, Hi);
            Put_Byte (Character'Pos (Lo)); Put_Byte (Character'Pos (Hi));

            Id := (if D /= 0.0 then 1.0 / D else 0.0);
            for J in 0 .. 31 loop
               declare
                  QV : Integer :=
                    Integer (Float'Rounding (Get_Flat (X, Base + J + 1) * Id)) + 16;
               begin
                  if QV < 0 then QV := 0; elsif QV > 31 then QV := 31; end if;
                  Q (J) := QV;
               end;
            end loop;

            --  qh: bit i (0..31) = 5th bit of element i.
            for J in 0 .. 31 loop
               if (Q (J) / 16) mod 2 = 1 then
                  QH := QH or Shift_Left (Unsigned_32 (1), J);
               end if;
            end loop;
            for K in 0 .. 3 loop
               Put_Byte (Natural (Shift_Right (QH, K * 8) and 16#FF#));
            end loop;

            --  qs: byte J holds lo4(q[J]) (low nibble) + lo4(q[J+16]) (high).
            for J in 0 .. 15 loop
               Put_Byte ((Q (J) mod 16) + 16 * (Q (J + 16) mod 16));
            end loop;
         end;
      end loop;
      return R;
   end Quantize_Q5_0;

   function Quantize_Q4_K (X : Tensor) return String is
      N  : constant Natural := Numel (X);
      SB : constant Natural := N / 256;
      R  : String (1 .. SB * 144);
      P  : Natural := 1;
      procedure Put_Byte (Bt : Natural) is
      begin R (P) := Character'Val (Bt mod 256); P := P + 1; end Put_Byte;
   begin
      for Block in 0 .. SB - 1 loop
         declare
            Base : constant Natural := Block * 256;
            A    : array (0 .. 7) of Float;    -- per-sub effective scale (>=0)
            NM   : array (0 .. 7) of Float;    -- per-sub effective neg-min (>=0)
            Sc_C : array (0 .. 7) of Natural := [others => 0];  -- 6-bit scale
            Mn_C : array (0 .. 7) of Natural := [others => 0];  -- 6-bit min
            Q    : array (0 .. 255) of Natural := [others => 0];
            D, DMin : Float := 0.0;
            AmaxA, AmaxM : Float := 0.0;
            Lo, Hi : Character;
            Sc : array (0 .. 11) of Natural := [others => 0];
         begin
            --  Per sub-block affine fit: out = A*q - NM, q in 0..15, offset <=0
            --  (so the neg-min code stays unsigned across the super-block).
            for S in 0 .. 7 loop
               declare
                  Mn : Float := Float'Last; Mx : Float := Float'First; Cmin : Float;
               begin
                  for L in 0 .. 31 loop
                     declare V : constant Float := Get_Flat (X, Base + S * 32 + L + 1);
                     begin
                        if V < Mn then Mn := V; end if;
                        if V > Mx then Mx := V; end if;
                     end;
                  end loop;
                  Cmin := Float'Min (Mn, 0.0);
                  A (S)  := (Mx - Cmin) / 15.0;
                  NM (S) := -Cmin;
               end;
            end loop;

            for S in 0 .. 7 loop
               if A (S)  > AmaxA then AmaxA := A (S);  end if;
               if NM (S) > AmaxM then AmaxM := NM (S); end if;
            end loop;
            D    := AmaxA / 63.0;
            DMin := AmaxM / 63.0;

            for S in 0 .. 7 loop
               if D    > 0.0 then Sc_C (S) := Natural (Float'Rounding (A (S)  / D));    end if;
               if DMin > 0.0 then Mn_C (S) := Natural (Float'Rounding (NM (S) / DMin)); end if;
               if Sc_C (S) > 63 then Sc_C (S) := 63; end if;
               if Mn_C (S) > 63 then Mn_C (S) := 63; end if;
            end loop;

            --  Quantize elements with the (reconstructable) effective scale/min.
            for S in 0 .. 7 loop
               declare
                  ES  : constant Float := D    * Float (Sc_C (S));   -- effective scale
                  EM  : constant Float := DMin * Float (Mn_C (S));   -- effective neg-min
                  Inv : constant Float := (if ES > 0.0 then 1.0 / ES else 0.0);
               begin
                  for L in 0 .. 31 loop
                     declare
                        X0 : constant Float := Get_Flat (X, Base + S * 32 + L + 1);
                        Qv : Integer := Integer (Float'Rounding ((X0 + EM) * Inv));
                     begin
                        if Qv < 0 then Qv := 0; elsif Qv > 15 then Qv := 15; end if;
                        Q (S * 32 + L) := Qv;
                     end;
                  end loop;
               end;
            end loop;

            --  d, dmin (f16), then the 12 packed 6-bit (scale,min) bytes.
            F32_To_F16 (D, Lo, Hi);
            Put_Byte (Character'Pos (Lo)); Put_Byte (Character'Pos (Hi));
            F32_To_F16 (DMin, Lo, Hi);
            Put_Byte (Character'Pos (Lo)); Put_Byte (Character'Pos (Hi));

            for J in 0 .. 3 loop
               Sc (J)     := Sc_C (J);     -- low 6 bits
               Sc (J + 4) := Mn_C (J);
            end loop;
            for J in 4 .. 7 loop
               Sc (J + 4) := (Sc_C (J) mod 16) + (Mn_C (J) mod 16) * 16;
               Sc (J - 4) := Sc (J - 4) + (Sc_C (J) / 16) * 64;   -- high 2 bits
               Sc (J)     := Sc (J)     + (Mn_C (J) / 16) * 64;
            end loop;
            for J in 0 .. 11 loop Put_Byte (Sc (J)); end loop;

            --  qs (128 bytes): byte (G*32+L) = q[2G*32+L] | q[(2G+1)*32+L] << 4.
            for G in 0 .. 3 loop
               for L in 0 .. 31 loop
                  Put_Byte (Q ((2 * G) * 32 + L) + 16 * Q ((2 * G + 1) * 32 + L));
               end loop;
            end loop;
         end;
      end loop;
      return R;
   end Quantize_Q4_K;

   function Quantize_Q5_K (X : Tensor) return String is
      N  : constant Natural := Numel (X);
      SB : constant Natural := N / 256;
      R  : String (1 .. SB * 176);
      P  : Natural := 1;
      procedure Put_Byte (Bt : Natural) is
      begin R (P) := Character'Val (Bt mod 256); P := P + 1; end Put_Byte;
   begin
      for Block in 0 .. SB - 1 loop
         declare
            Base : constant Natural := Block * 256;
            A    : array (0 .. 7) of Float;    -- per-sub effective scale (>=0)
            NM   : array (0 .. 7) of Float;    -- per-sub effective neg-min (>=0)
            Sc_C : array (0 .. 7) of Natural := [others => 0];  -- 6-bit scale
            Mn_C : array (0 .. 7) of Natural := [others => 0];  -- 6-bit min
            Q    : array (0 .. 255) of Natural := [others => 0]; -- 5-bit (0..31)
            D, DMin : Float := 0.0;
            AmaxA, AmaxM : Float := 0.0;
            Lo, Hi : Character;
            Sc : array (0 .. 11) of Natural := [others => 0];
            QH : array (0 .. 31)  of Natural := [others => 0];
         begin
            --  Per 32-element sub-block affine fit: out = A*q - NM, q in 0..31.
            for S in 0 .. 7 loop
               declare
                  Mn : Float := Float'Last; Mx : Float := Float'First; Cmin : Float;
               begin
                  for L in 0 .. 31 loop
                     declare V : constant Float := Get_Flat (X, Base + S * 32 + L + 1);
                     begin
                        if V < Mn then Mn := V; end if;
                        if V > Mx then Mx := V; end if;
                     end;
                  end loop;
                  Cmin := Float'Min (Mn, 0.0);
                  A (S)  := (Mx - Cmin) / 31.0;     -- 5-bit grid
                  NM (S) := -Cmin;
               end;
            end loop;

            for S in 0 .. 7 loop
               if A (S)  > AmaxA then AmaxA := A (S);  end if;
               if NM (S) > AmaxM then AmaxM := NM (S); end if;
            end loop;
            D    := AmaxA / 63.0;
            DMin := AmaxM / 63.0;

            for S in 0 .. 7 loop
               if D    > 0.0 then Sc_C (S) := Natural (Float'Rounding (A (S)  / D));    end if;
               if DMin > 0.0 then Mn_C (S) := Natural (Float'Rounding (NM (S) / DMin)); end if;
               if Sc_C (S) > 63 then Sc_C (S) := 63; end if;
               if Mn_C (S) > 63 then Mn_C (S) := 63; end if;
            end loop;

            for S in 0 .. 7 loop
               declare
                  ES  : constant Float := D    * Float (Sc_C (S));
                  EM  : constant Float := DMin * Float (Mn_C (S));
                  Inv : constant Float := (if ES > 0.0 then 1.0 / ES else 0.0);
               begin
                  for L in 0 .. 31 loop
                     declare
                        X0 : constant Float := Get_Flat (X, Base + S * 32 + L + 1);
                        Qv : Integer := Integer (Float'Rounding ((X0 + EM) * Inv));
                     begin
                        if Qv < 0 then Qv := 0; elsif Qv > 31 then Qv := 31; end if;
                        Q (S * 32 + L) := Qv;
                     end;
                  end loop;
               end;
            end loop;

            --  d, dmin (f16), then the 12 packed 6-bit (scale,min) bytes
            --  (identical packing to Q4_K / get_scale_min_k4).
            F32_To_F16 (D, Lo, Hi);
            Put_Byte (Character'Pos (Lo)); Put_Byte (Character'Pos (Hi));
            F32_To_F16 (DMin, Lo, Hi);
            Put_Byte (Character'Pos (Lo)); Put_Byte (Character'Pos (Hi));

            for J in 0 .. 3 loop
               Sc (J)     := Sc_C (J);
               Sc (J + 4) := Mn_C (J);
            end loop;
            for J in 4 .. 7 loop
               Sc (J + 4) := (Sc_C (J) mod 16) + (Mn_C (J) mod 16) * 16;
               Sc (J - 4) := Sc (J - 4) + (Sc_C (J) / 16) * 64;
               Sc (J)     := Sc (J)     + (Mn_C (J) / 16) * 64;
            end loop;
            for J in 0 .. 11 loop Put_Byte (Sc (J)); end loop;

            --  qh (32 bytes): QH[L] bit J = 5th bit of element J*32+L.
            for L in 0 .. 31 loop
               declare Bt : Natural := 0;
               begin
                  for J in 0 .. 7 loop
                     Bt := Bt + ((Q (J * 32 + L) / 16) mod 2) * (2 ** J);
                  end loop;
                  QH (L) := Bt;
               end;
            end loop;
            for L in 0 .. 31 loop Put_Byte (QH (L)); end loop;

            --  qs (128 bytes): byte (G*32+L) = lo4(q[2G*32+L]) | lo4(q[(2G+1)*32+L])<<4.
            for G in 0 .. 3 loop
               for L in 0 .. 31 loop
                  Put_Byte ((Q ((2 * G) * 32 + L) mod 16)
                            + 16 * (Q ((2 * G + 1) * 32 + L) mod 16));
               end loop;
            end loop;
         end;
      end loop;
      return R;
   end Quantize_Q5_K;

   function Quantize_Q6_K (X : Tensor) return String is
      N  : constant Natural := Numel (X);
      SB : constant Natural := N / 256;
      R  : String (1 .. SB * 210);
      P  : Natural := 1;
      procedure Put_Byte (Bt : Natural) is
      begin R (P) := Character'Val (Bt mod 256); P := P + 1; end Put_Byte;
   begin
      for Block in 0 .. SB - 1 loop
         declare
            Base : constant Natural := Block * 256;
            GS   : array (0 .. 15) of Float := [others => 0.0];  -- ideal grp scale
            ES   : array (0 .. 15) of Float := [others => 0.0];  -- reconstructable
            Sc8  : array (0 .. 15) of Integer := [others => 0];  -- int8 scale codes
            Q    : array (0 .. 255) of Natural := [others => 32]; -- 6-bit (0..63)
            As, D  : Float := 0.0;
            Lo, Hi : Character;
            QL : array (0 .. 127) of Natural := [others => 0];
            QH : array (0 .. 63)  of Natural := [others => 0];
         begin
            --  Per 16-element group: signed-max scale (the max-abs element maps
            --  to q-32 = -32, i.e. reconstructed exactly), like Q4_0's d=v/-8.
            for G in 0 .. 15 loop
               declare
                  Amax : Float := 0.0; V : Float := 0.0;
               begin
                  for L in 0 .. 15 loop
                     declare F : constant Float := Get_Flat (X, Base + G * 16 + L + 1);
                     begin if abs F > Amax then Amax := abs F; V := F; end if; end;
                  end loop;
                  GS (G) := (if Amax > 0.0 then V / (-32.0) else 0.0);
               end;
            end loop;

            --  Shared f16 d encodes the 16 signed group scales as int8.
            for G in 0 .. 15 loop
               if abs GS (G) > As then As := abs GS (G); end if;
            end loop;
            D := As / 127.0;
            for G in 0 .. 15 loop
               if D > 0.0 then
                  Sc8 (G) := Integer (Float'Rounding (GS (G) / D));
                  if Sc8 (G) > 127 then Sc8 (G) := 127;
                  elsif Sc8 (G) < -128 then Sc8 (G) := -128; end if;
               end if;
               ES (G) := D * Float (Sc8 (G));          -- what the decoder rebuilds
            end loop;

            --  Quantize each element against the reconstructable scale.
            for G in 0 .. 15 loop
               declare
                  Inv : constant Float := (if ES (G) /= 0.0 then 1.0 / ES (G) else 0.0);
               begin
                  for L in 0 .. 15 loop
                     declare
                        Qi : Integer :=
                          Integer (Float'Rounding (Get_Flat (X, Base + G * 16 + L + 1) * Inv));
                     begin
                        if Qi < -32 then Qi := -32; elsif Qi > 31 then Qi := 31; end if;
                        Q (G * 16 + L) := Qi + 32;      -- 0..63
                     end;
                  end loop;
               end;
            end loop;

            --  Pack ql/qh to mirror Decode_Q6K_Block. Per half (128 elems):
            --  L in 0..31 indexes four streams at half-offsets 0/32/64/96.
            for Half in 0 .. 1 loop
               declare
                  QLO : constant Natural := Half * 64;   -- ql byte base
                  QHO : constant Natural := Half * 32;   -- qh byte base
                  YH  : constant Natural := Half * 128;  -- element base
               begin
                  for L in 0 .. 31 loop
                     declare
                        Q1 : constant Natural := Q (YH + L);
                        Q2 : constant Natural := Q (YH + L + 32);
                        Q3 : constant Natural := Q (YH + L + 64);
                        Q4 : constant Natural := Q (YH + L + 96);
                     begin
                        QL (QLO + L)      := (Q1 mod 16) + (Q3 mod 16) * 16;
                        QL (QLO + L + 32) := (Q2 mod 16) + (Q4 mod 16) * 16;
                        QH (QHO + L) :=
                          ((Q1 / 16) mod 4)
                          + ((Q2 / 16) mod 4) * 4
                          + ((Q3 / 16) mod 4) * 16
                          + ((Q4 / 16) mod 4) * 64;
                     end;
                  end loop;
               end;
            end loop;

            --  Emit: ql[128], qh[32], scales[16] (int8), d (f16) — 210 bytes.
            for J in 0 .. 127 loop Put_Byte (QL (J)); end loop;
            for J in 0 .. 63  loop Put_Byte (QH (J)); end loop;
            for J in 0 .. 15  loop
               Put_Byte (if Sc8 (J) < 0 then Sc8 (J) + 256 else Sc8 (J));
            end loop;
            F32_To_F16 (D, Lo, Hi);
            Put_Byte (Character'Pos (Lo)); Put_Byte (Character'Pos (Hi));
         end;
      end loop;
      return R;
   end Quantize_Q6_K;

end LLM_Quant;
