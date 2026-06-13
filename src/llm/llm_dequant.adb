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

   function F16_To_F32 (X : Byte; Y : Byte) return Float is
      Sign     : constant Integer := Integer (X and 128);
      Exponent : constant Integer := Integer (Shift_Right (X and 124, 2));
      Mantissa : constant Integer := Integer (Shift_Left (Unsigned_16 (X and 3), 8)) + Integer (Y);
      Value    : Float;
   begin
      if Exponent = 0 then
         -- Subnormal or zero
         if Mantissa = 0 then
            return (if Sign = 0 then 0.0 else -0.0);
         end if;
         Value := Float (Mantissa) / 1024.0;
      elsif Exponent = 31 then
         -- Infinity or NaN
         return 0.0;
      else
         Value := Float (Mantissa + 1024) / 1024.0;
         Value := Value * Float (2 ** (Exponent - 15));
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

   procedure Dequant_Q8_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;
   begin
      for B in 1 .. Blocks loop
         declare
            D : constant Float := F16_To_F32 (
              Byte (Character'Pos (X (Pos))),
              Byte (Character'Pos (X (Pos + 1))));
            QS : array (1 .. 256) of Byte;
         begin
            Pos := Pos + 2;
            for I in 1 .. 256 loop
               QS (I) := Byte (Character'Pos (X (Pos + I - 1)));
            end loop;
            Pos := Pos + 256;

            for I in 1 .. 256 loop
               Set_Flat (Q, Q_Pos, D * Float (QS (I)));
               Q_Pos := Q_Pos + 1;
            end loop;
         end;
      end loop;
   end Dequant_Q8_K;

   --------------------------------------------------------------------
   -- Q6_K dequantization
   --
   -- Super-block of 256 elements (Q6_K_M variant):
   --   - 2 bytes: FP16 d (global scale for super-block)
   --   - 128 bytes: 16 × FP16 scales (one per 16-element sub-block)
   --   - 16 bytes: 16 × int8 mins  (one per 16-element sub-block)
   --   - 256 * 6/8 = 192 bytes: qs (packed 6-bit, 4 values per 3 bytes)
   --   - 32 bytes: qh (high 2 bits, 1 byte per 16 el, 4×2bit)
   -- Total: 2 + 128 + 16 + 192 + 32 = 370 bytes / 256 el
   --
   -- Reference: llama.cpp ggml_quants.c, dequantize_row_q6_K
   --------------------------------------------------------------------

   procedure Dequant_Q6_K (X : String; Q : out Tensor; N : Natural) is
      Blocks : constant Natural := N / 256;
      Pos    : Natural := X'First;
      Q_Pos  : Natural := 1;
   begin
      for B in 1 .. Blocks loop
         declare
            D      : constant Float := F16_To_F32 (
              Byte (Character'Pos (X (Pos))),
              Byte (Character'Pos (X (Pos + 1))));
            Scales : array (1 .. 16) of Float;
            QS     : array (1 .. 192) of Byte;
            QH     : array (1 .. 32) of Byte;
         begin
            Pos := Pos + 2;

            -- 16 × FP16 sub-block scales
            for I in 1 .. 16 loop
               Scales (I) := F16_To_F32 (
                 Byte (Character'Pos (X (Pos))),
                 Byte (Character'Pos (X (Pos + 1))));
               Pos := Pos + 2;
            end loop;

            -- 16 × int8 mins (stored as bytes): not used by this dequant
            -- path, so skip over the 16-byte region without decoding.
            Pos := Pos + 16;

            -- qs: 192 bytes (256 elements × 6 bit / 8 = 192)
            for I in 1 .. 192 loop
               QS (I) := Byte (Character'Pos (X (Pos + I - 1)));
            end loop;
            Pos := Pos + 192;

            -- qh: 32 bytes (additional 2 bits for each 6-bit, nibble per 16)
            for I in 1 .. 32 loop
               QH (I) := Byte (Character'Pos (X (Pos + I - 1)));
            end loop;
            Pos := Pos + 32;

            -- Dequant: 16 sub-blocks of 16 elements
            for SB in 1 .. 16 loop
               for I in 1 .. 16 loop
                  declare
                     El_Idx  : constant Natural := (SB - 1) * 16 + I;
                     Byte_Idx : constant Natural := (El_Idx - 1) * 6 / 8 + 1;
                     Bit_Off  : constant Natural := (El_Idx - 1) * 6 mod 8;
                     Low      : constant Integer := Integer (Shift_Right (QS (Byte_Idx), Bit_Off) and 63);
                     High_Nibble : constant Integer := Integer (Shift_Right (QH (SB), 4));  -- upper 2 bits
                     High     : constant Integer := Integer (
                       Shift_Left (Unsigned_32 (High_Nibble), 2));
                     QVal     : constant Integer := Low + High;
                  begin
                     Set_Flat (Q, Q_Pos, D * Scales (SB) * Float (QVal - 32));
                     Q_Pos := Q_Pos + 1;
                  end;
               end loop;
            end loop;
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

   procedure Dequant_Q5_K (X : String; Q : out Tensor; N : Natural) is
      Blocks      : constant Natural := N / 256;
      Pos         : Natural := X'First;
      Q_Pos       : Natural := 1;
   begin
      for B in 1 .. Blocks loop
         declare
            Scales : array (1 .. 12) of Float;
            QS     : array (1 .. 32) of Byte;
            QH     : array (1 .. 16) of Byte;
         begin
            -- Read 12 FP16 scales
            for I in 1 .. 12 loop
               Scales (I) := F16_To_F32 (
                 Byte (Character'Pos (X (Pos))),
                 Byte (Character'Pos (X (Pos + 1))));
               Pos := Pos + 2;
            end loop;

            -- Read qs: (256/8) = 32 bytes of packed 5-bit values
            for I in 1 .. 32 loop
               QS (I) := Byte (Character'Pos (X (Pos + I - 1)));
            end loop;
            Pos := Pos + 32;

            -- Read qh: (256/16) = 16 bytes of high bits
            for I in 1 .. 16 loop
               QH (I) := Byte (Character'Pos (X (Pos + I - 1)));
            end loop;
            Pos := Pos + 16;

            -- Dequantize 256 values in this block
            for I in 1 .. 256 loop
               declare
                  Sub_Block : constant Natural := (I - 1) / 32 + 1;
                  Low       : constant Byte := Shift_Right (QS ((I - 1) / 8 + 1), (I - 1) mod 8);
                  Low_Val   : constant Integer := Integer (Low and 15);

                  High_Byte_Pos : constant Natural := (I - 1) / 16 + 1;
                  High_Bit_Pos  : constant Natural := (I - 1) mod 16;
                  High_Bit      : constant Integer := Integer (Shift_Right (QH (High_Byte_Pos), High_Bit_Pos) and 1);

                  QVal  : constant Integer := Low_Val + Integer (Shift_Left (Unsigned_32 (High_Bit), 4));
                  Scale : Float;
                  DMin  : Float;
               begin
                  if Sub_Block <= 4 then
                     Scale := Scales (2 * Sub_Block);
                     DMin  := Scales (2 * Sub_Block - 1);
                  else
                     Scale := Scales (2 * (Sub_Block - 4) + 8);
                     DMin  := Scales (2 * (Sub_Block - 4) + 7);
                  end if;

                  Set_Flat (Q, Q_Pos, Scale * Float (QVal - 16) + DMin);
                  Q_Pos := Q_Pos + 1;
               end;
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

   function Dequantize
     (Info : LLM_GGUF.Tensor_Info;
      Raw  : String)
      return Tensor
   is
      N : constant Natural := Dequant_Num_Elements (Info);
      Result : Tensor := New_Tensor ([1, N]);
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
