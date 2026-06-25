------------------------------------------------------------------------
-- test_kquant_synth — self-contained (no model) regression guard for the
-- Q2_K and Q3_K decoders, which otherwise are only exercised by skip-if-absent
-- real-model tests. Hand-builds one super-block per format from chosen quant
-- codes + per-16 scales, laid out byte-for-byte per the ggml struct, then:
--   (1) Dequantize must reproduce the values computed directly from the ggml
--       formula (decode correctness — offsets, field/scale interleave, f16);
--   (2) QMatVec must equal a manual dot of the dequantized row (fused path).
-- d/dmin are exact in f16 (0.5 / 0.25) so the expected values are exact.
-- The decoders are independently validated against real models; this freezes
-- that known-correct behaviour so a future refactor can't break it silently.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_GGUF;
with LLM_Dequant;
with LLM_Quant;
with LLM_Tensor;            use LLM_Tensor;

procedure Test_KQuant_Synth is
   Pass : Boolean := True;
   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   type Byte_Arr is array (Natural range <>) of Natural;
   type Elem_Arr is array (0 .. 255) of Natural;
   type Val_Arr  is array (0 .. 255) of Float;
   type Sc_Arr   is array (0 .. 15)  of Natural;

   D    : constant Float := 0.5;    -- exact in f16
   DMin : constant Float := 0.25;   -- exact in f16

   --  Element P -> its position in the shared K-quant interleave.
   procedure Map (P : Natural; QByte, Shift, Iss, HByte, HBit : out Natural) is
      NH : constant Natural := P / 128;
      Pp : constant Natural := P mod 128;
      J  : constant Natural := Pp / 32;
      Rm : constant Natural := Pp mod 32;
      G  : constant Natural := Rm / 16;
      L  : constant Natural := Rm mod 16;
   begin
      QByte := NH * 32 + G * 16 + L;   -- qs byte (0..63)
      Shift := 2 * J;                  -- 2-bit field within that byte
      Iss   := NH * 8 + J * 2 + G;     -- scale index (0..15)
      HByte := G * 16 + L;             -- hmask byte (Q3_K, 0..31)
      HBit  := NH * 4 + J;             -- hmask bit  (Q3_K, 0..7)
   end Map;

   function To_Raw (B : Byte_Arr) return String is
      R : String (1 .. B'Length);
   begin
      for I in B'Range loop
         R (I - B'First + 1) := Character'Val (B (I) mod 256);
      end loop;
      return R;
   end To_Raw;

   procedure Put_F16 (B : in out Byte_Arr; At_I : Natural; V : Float) is
      Lo, Hi : Character;
   begin
      LLM_Quant.F32_To_F16 (V, Lo, Hi);
      B (At_I)     := Character'Pos (Lo);
      B (At_I + 1) := Character'Pos (Hi);
   end Put_F16;

   --  Max |Dequantize(Raw) - Expected| over the 256 elements.
   function Deq_Err (Info : LLM_GGUF.Tensor_Info; Raw : String; Exp : Val_Arr)
                     return Float
   is
      Q : constant Tensor := LLM_Dequant.Dequantize (Info, Raw);
      M : Float := 0.0;
   begin
      for P in 0 .. 255 loop
         M := Float'Max (M, abs (Get_Flat (Q, P + 1) - Exp (P)));
      end loop;
      return M;
   end Deq_Err;

   --  |QMatVec(W,x) - sum(dequant(W)*x)| for a single-row weight.
   function QMV_Err (Info : LLM_GGUF.Tensor_Info; Raw : String) return Float is
      X    : Tensor := New_Tensor ([1, 256]);
      Wrow : constant Tensor := LLM_Dequant.Dequantize (Info, Raw);
      Dot  : Float := 0.0;
   begin
      for I in 1 .. 256 loop
         Set_Flat (X, I, 0.01 * Float (((I mod 11) - 5)));
      end loop;
      for I in 1 .. 256 loop
         Dot := Dot + Get_Flat (Wrow, I) * Get_Flat (X, I);
      end loop;
      return abs (Get_Flat (LLM_Dequant.QMatVec (Info, Raw, X), 1) - Dot);
   end QMV_Err;

   function Info_Of (Kind : LLM_GGUF.GGML_Type) return LLM_GGUF.Tensor_Info is
     (Name => Null_Unbounded_String, N_Dims => 2, Dims => [256, 1, 0, 0],
      Kind => Kind, Offset => 0, Byte_Size => 0);

begin
   Put_Line ("=== synthetic Q2_K / Q3_K block decode ===");

   --------------------------------------------------------------------
   --  Q2_K: 84 B = scales[16] (4-bit scale low | 4-bit min high) + qs[64]
   --  (2-bit) + d (f16) + dmin (f16). y = d*scale4*q2 - dmin*min4.
   --------------------------------------------------------------------
   New_Line; Put_Line ("--- Q2_K ---");
   declare
      Blk    : Byte_Arr (0 .. 83) := [others => 0];
      Scale4 : Sc_Arr;  Min4 : Sc_Arr;
      Q2     : Elem_Arr;  Exp : Val_Arr;
      Info   : constant LLM_GGUF.Tensor_Info := Info_Of (LLM_GGUF.GGML_TYPE_Q2_K);
   begin
      for S in 0 .. 15 loop
         Scale4 (S) := (S mod 7) + 1;   -- 1..7
         Min4 (S)   := S mod 5;         -- 0..4
         Blk (S)    := Scale4 (S) + 16 * Min4 (S);
      end loop;
      for P in 0 .. 255 loop
         Q2 (P) := P mod 4;             -- exercise all 4 codes & positions
         declare QB, Sh, Iss, HB, HBit : Natural; begin
            Map (P, QB, Sh, Iss, HB, HBit);
            Blk (16 + QB) := Blk (16 + QB) + Q2 (P) * (2 ** Sh);
            Exp (P) := D * Float (Scale4 (Iss)) * Float (Q2 (P))
                       - DMin * Float (Min4 (Iss));
         end;
      end loop;
      Put_F16 (Blk, 80, D);
      Put_F16 (Blk, 82, DMin);

      declare Raw : constant String := To_Raw (Blk); begin
         Check ("Q2_K decode matches ggml formula", Deq_Err (Info, Raw, Exp) < 1.0e-6);
         Check ("Q2_K fused QMatVec == dense dot",  QMV_Err (Info, Raw) < 1.0e-4);
      end;
   end;

   --------------------------------------------------------------------
   --  Q3_K: 110 B = hmask[32] + qs[64] (2-bit) + scales[12] (16 signed
   --  6-bit, packed) + d (f16). y = d*(scale6-32)*((low2|hbit<<2)-4).
   --------------------------------------------------------------------
   New_Line; Put_Line ("--- Q3_K ---");
   declare
      Blk    : Byte_Arr (0 .. 109) := [others => 0];
      Scale6 : Sc_Arr;
      Q3     : Elem_Arr;  Exp : Val_Arr;
      Info   : constant LLM_GGUF.Tensor_Info := Info_Of (LLM_GGUF.GGML_TYPE_Q3_K);
   begin
      for S in 0 .. 15 loop
         Scale6 (S) := (S * 4) mod 64;   -- 0..60 -> scale-32 spans -32..28
      end loop;
      --  Pack the 16 six-bit scales into 12 bytes (inverse of the decoder's
      --  kmask1/kmask2 expansion): bytes 0-3 = A0, 4-7 = A1, 8-11 = Tmp.
      for K in 0 .. 3 loop
         Blk (96 + K)     := (Scale6 (K)     mod 16) + (Scale6 (8 + K)  mod 16) * 16;
         Blk (96 + 4 + K) := (Scale6 (4 + K) mod 16) + (Scale6 (12 + K) mod 16) * 16;
         Blk (96 + 8 + K) := (Scale6 (K) / 16)
                             + (Scale6 (4 + K)  / 16) * 4
                             + (Scale6 (8 + K)  / 16) * 16
                             + (Scale6 (12 + K) / 16) * 64;
      end loop;
      for P in 0 .. 255 loop
         Q3 (P) := P mod 8;             -- all 3-bit codes (low2 + high bit)
         declare QB, Sh, Iss, HB, HBit : Natural; begin
            Map (P, QB, Sh, Iss, HB, HBit);
            Blk (32 + QB) := Blk (32 + QB) + (Q3 (P) mod 4) * (2 ** Sh);  -- qs low 2
            Blk (HB)      := Blk (HB) + (Q3 (P) / 4) * (2 ** HBit);        -- hmask hi bit
            Exp (P) := D * Float (Integer (Scale6 (Iss)) - 32)
                          * Float (Integer (Q3 (P)) - 4);
         end;
      end loop;
      Put_F16 (Blk, 108, D);

      declare Raw : constant String := To_Raw (Blk); begin
         Check ("Q3_K decode matches ggml formula", Deq_Err (Info, Raw, Exp) < 1.0e-6);
         Check ("Q3_K fused QMatVec == dense dot",  QMV_Err (Info, Raw) < 1.0e-4);
      end;
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_KQuant_Synth;
