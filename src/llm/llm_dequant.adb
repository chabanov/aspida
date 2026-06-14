---------------------------------------------------------------------
-- LLM_Dequant body — GGML quantization → FP32 conversion
---------------------------------------------------------------------

with Ada.Text_IO;
with Interfaces; use Interfaces;

package body LLM_Dequant is

   use LLM_Tensor;

   subtype Byte is Interfaces.Unsigned_8;

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

   procedure Dequant_Q6_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Pos    : Natural := X'First;
      Q_Base : Natural := 1;   -- 1-based flat index of this block's first elem

      function S8 (P : Natural) return Integer is   -- signed int8 at byte P
         U : constant Integer := Character'Pos (X (P));
      begin
         return (if U >= 128 then U - 256 else U);
      end S8;
   begin
      for B in 1 .. Blocks loop
         declare
            QL_Base : constant Natural := Pos;          -- ql[128]
            QH_Base : constant Natural := Pos + 128;    -- qh[64]
            SC_Base : constant Natural := Pos + 192;    -- scales[16]
            D       : constant Float := F16_To_F32 (
              Byte (Character'Pos (X (Pos + 208))),
              Byte (Character'Pos (X (Pos + 209))));
         begin
            for Half in 0 .. 1 loop
               declare
                  QL_H : constant Natural := QL_Base + Half * 64;
                  QH_H : constant Natural := QH_Base + Half * 32;
                  SC_H : constant Natural := SC_Base + Half * 8;
                  Y_H  : constant Natural := Q_Base + Half * 128;
               begin
                  for L in 0 .. 31 loop
                     declare
                        Is_Idx : constant Natural := L / 16;  -- 0 or 1
                        QL_L   : constant Unsigned_32 :=
                          Unsigned_32 (Character'Pos (X (QL_H + L)));
                        QL_L32 : constant Unsigned_32 :=
                          Unsigned_32 (Character'Pos (X (QL_H + L + 32)));
                        QH_L   : constant Unsigned_32 :=
                          Unsigned_32 (Character'Pos (X (QH_H + L)));
                        Q1 : constant Integer :=
                          Integer ((QL_L and 16#0F#)
                                   or Shift_Left (QH_L and 3, 4)) - 32;
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
                        Set_Flat (Q, Y_H + L,
                          D * Float (S8 (SC_H + Is_Idx + 0)) * Float (Q1));
                        Set_Flat (Q, Y_H + L + 32,
                          D * Float (S8 (SC_H + Is_Idx + 2)) * Float (Q2));
                        Set_Flat (Q, Y_H + L + 64,
                          D * Float (S8 (SC_H + Is_Idx + 4)) * Float (Q3));
                        Set_Flat (Q, Y_H + L + 96,
                          D * Float (S8 (SC_H + Is_Idx + 6)) * Float (Q4));
                     end;
                  end loop;
               end;
            end loop;
            Pos := Pos + 210;
            Q_Base := Q_Base + 256;
         end;
      end loop;
   end Dequant_Q6_K;

   --------------------------------------------------------------------
   -- Q4_K dequantization
   --
   -- Super-block of 256 elements (Q4_K_M variant):
   --   - 2 bytes: FP16 d (global scale)
   --   - 2 bytes: FP16 dmin (global min scale)
   --   - 24 bytes: 12 × FP16 scales (6 sub-blocks of 32 el, each has scale + min)
   --   - 128 bytes: qs (packed 4-bit, 2 values per byte)
   --   - 16 bytes: qh (high 2 bits, 1 byte per 32 el)
   -- Total: 2+2+24+128+16 = 172 bytes / 256 el
   --
   -- Sub-blocks of 32: 8 groups within super-block
   -- scales[2*i]=sub_scale, scales[2*i-1]=sub_min for i=1..6
   --
   -- Reference: llama.cpp, dequantize_row_q4_K
   --------------------------------------------------------------------

   procedure Dequant_Q4_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;
   begin
      for B in 1 .. Blocks loop
         declare
            D     : constant Float := F16_To_F32 (
              Byte (Character'Pos (X (Pos))),
              Byte (Character'Pos (X (Pos + 1))));
            DMin  : constant Float := F16_To_F32 (
              Byte (Character'Pos (X (Pos + 2))),
              Byte (Character'Pos (X (Pos + 3))));
            Scales : array (1 .. 12) of Float;
            QS     : array (1 .. 128) of Byte;
            QH     : array (1 .. 16) of Byte;
         begin
            Pos := Pos + 4;

            -- 12 × FP16 sub-block scales+mins
            for I in 1 .. 12 loop
               Scales (I) := F16_To_F32 (
                 Byte (Character'Pos (X (Pos))),
                 Byte (Character'Pos (X (Pos + 1))));
               Pos := Pos + 2;
            end loop;

            -- qs: 128 bytes (256 × 4 bit / 8)
            for I in 1 .. 128 loop
               QS (I) := Byte (Character'Pos (X (Pos + I - 1)));
            end loop;
            Pos := Pos + 128;

            -- qh: 16 bytes (additional 2 bits, 1 byte per 16 el = nibble pair)
            for I in 1 .. 16 loop
               QH (I) := Byte (Character'Pos (X (Pos + I - 1)));
            end loop;
            Pos := Pos + 16;

            -- Dequant: 256 elements, grouped into sub-blocks of 32
            for SB in 1 .. 8 loop
               declare
                  Sub_Scale : constant Float := D * Scales (2 * ((SB + 2) / 4) * 4 + 1 - (SB mod 4) * 2);
                  Sub_Min   : constant Float := DMin * Scales (2 * ((SB + 2) / 4) * 4 - (SB mod 4) * 2);
               begin
                  for I in 1 .. 32 loop
                     declare
                        El_Idx    : constant Natural := (SB - 1) * 32 + I;
                        Nibble_Idx : constant Natural := (El_Idx - 1) / 2 + 1;
                        Low_Nibble : constant Integer := Integer (QS (Nibble_Idx) and 15);
                        High_Nibble_Hi : constant Integer := Integer (Shift_Right (QH ((SB - 1) * 2 + 1), 4));
                        High_Nibble_Lo : constant Integer := Integer (Shift_Right (QH ((SB - 1) * 2 + 2), 4));
                        QVal : Integer;
                     begin
                        if I <= 16 then
                           QVal := Low_Nibble + High_Nibble_Hi * 16;
                        else
                           QVal := Low_Nibble + High_Nibble_Lo * 16;
                        end if;
                        Set_Flat (Q, Q_Pos, Sub_Scale * Float (QVal) + Sub_Min);
                        Q_Pos := Q_Pos + 1;
                     end;
                  end loop;
               end;
            end loop;
         end;
      end loop;
   end Dequant_Q4_K;
   --
   -- N = block size (typically 256)
   -- Blocks = N / 256
   -- Each block of 256:
   --   - 24 bytes of scales (12 × FP16 → 12 values in FP32)
   --   - (N/8) bytes of qs (compressed 5-bit values, 8 per byte)
   --   - (N/16) bytes of qh (high bits, 1 per 2 values for 5th bit)
   --------------------------------------------------------------------

   --  Q5_K super-block (llama.cpp block_q5_K, 176 bytes / 256 elements):
   --    ggml_half d;        -- f16 super-block scale for the 6-bit scales
   --    ggml_half dmin;     -- f16 super-block scale for the 6-bit mins
   --    uint8_t scales[12]; -- 8 sub-block scales + 8 mins, 6-bit packed
   --    uint8_t qh[32];     -- 5th (high) bit of each quant
   --    uint8_t qs[128];    -- low 4 bits, 2 quants per byte
   --  Value: y = d*sc*((qs&0xF | qh_bit<<4)) - dmin*m, per 32-element sub-block.
   procedure Dequant_Q5_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;

      function RF16 (P : Natural) return Float is
        (F16_To_F32 (Byte (Character'Pos (X (P))),
                     Byte (Character'Pos (X (P + 1)))));
   begin
      for B in 1 .. Blocks loop
         declare
            D    : constant Float := RF16 (Pos);
            DMin : constant Float := RF16 (Pos + 2);
            Sc   : array (0 .. 11) of Unsigned_32;
            QH   : array (0 .. 31) of Unsigned_32;
            QS   : array (0 .. 127) of Unsigned_32;

            --  get_scale_min_k4: unpack the j-th 6-bit scale and min.
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

            Is_Idx : Natural := 0;
            U1     : Unsigned_32 := 1;
            U2     : Unsigned_32 := 2;
            QS_Off : Natural := 0;
         begin
            Pos := Pos + 4;
            for I in 0 .. 11 loop
               Sc (I) := Unsigned_32 (Character'Pos (X (Pos + I)));
            end loop;
            Pos := Pos + 12;
            for I in 0 .. 31 loop
               QH (I) := Unsigned_32 (Character'Pos (X (Pos + I)));
            end loop;
            Pos := Pos + 32;
            for I in 0 .. 127 loop
               QS (I) := Unsigned_32 (Character'Pos (X (Pos + I)));
            end loop;
            Pos := Pos + 128;

            --  4 groups of 64 elements (two 32-element sub-blocks each).
            for G in 0 .. 3 loop
               declare
                  Sc1, M1, Sc2, M2 : Unsigned_32;
               begin
                  Scale_Min (Is_Idx,     Sc1, M1);
                  Scale_Min (Is_Idx + 1, Sc2, M2);
                  declare
                     D1  : constant Float := D * Float (Sc1);
                     Mn1 : constant Float := DMin * Float (M1);
                     D2  : constant Float := D * Float (Sc2);
                     Mn2 : constant Float := DMin * Float (M2);
                  begin
                     for L in 0 .. 31 loop
                        declare
                           Hi : constant Unsigned_32 :=
                             (if (QH (L) and U1) /= 0 then 16 else 0);
                           Qv : constant Unsigned_32 := (QS (QS_Off + L) and 16#0F#) + Hi;
                        begin
                           Set_Flat (Q, Q_Pos, D1 * Float (Qv) - Mn1);
                           Q_Pos := Q_Pos + 1;
                        end;
                     end loop;
                     for L in 0 .. 31 loop
                        declare
                           Hi : constant Unsigned_32 :=
                             (if (QH (L) and U2) /= 0 then 16 else 0);
                           Qv : constant Unsigned_32 := Shift_Right (QS (QS_Off + L), 4) + Hi;
                        begin
                           Set_Flat (Q, Q_Pos, D2 * Float (Qv) - Mn2);
                           Q_Pos := Q_Pos + 1;
                        end;
                     end loop;
                  end;
               end;
               QS_Off := QS_Off + 32;
               Is_Idx := Is_Idx + 2;
               U1 := Shift_Left (U1, 2);
               U2 := Shift_Left (U2, 2);
            end loop;
         end;
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

   function Dequantize
     (Info : LLM_GGUF.Tensor_Info;
      Raw  : String)
      return Tensor
   is
      N : constant Natural := Dequant_Num_Elements (Info);
      Result : Tensor := New_Tensor (Logical_Shape (Info));
   begin
      case Info.Kind is
         when LLM_GGUF.GGML_TYPE_F32 =>
            -- Direct copy: each 4 bytes → 1 float
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

         when LLM_GGUF.GGML_TYPE_Q5_K =>
            Dequant_Q5_K (Raw, Result, N);

         when LLM_GGUF.GGML_TYPE_Q8_K =>
            Dequant_Q8_K (Raw, Result, N);

         when LLM_GGUF.GGML_TYPE_Q6_K =>
            Dequant_Q6_K (Raw, Result, N);

         when LLM_GGUF.GGML_TYPE_Q4_K =>
            Dequant_Q4_K (Raw, Result, N);

         when others =>
            Ada.Text_IO.Put_Line ("Dequantize: unsupported type " &
              LLM_GGUF.GGML_Type'Image (Info.Kind));
            -- Zero-initialized already
            null;
      end case;
      return Result;
   end Dequantize;

end LLM_Dequant;
