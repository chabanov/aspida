------------------------------------------------------------------------
-- test_quant — FP32 -> Q8_0 quantizer round-trips through the engine's
-- dequantizer within the Q8_0 error bound (~half a quant step). Validates the
-- quantizer against the independent, already-correct LLM_Dequant path.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_GGUF;
with LLM_Dequant;
with LLM_Quant;
with LLM_Tensor;            use LLM_Tensor;

procedure Test_Quant is
   Pass : Boolean := True;
   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   N : constant := 64;   -- 2 Q8_0 blocks

   function Info return LLM_GGUF.Tensor_Info is
     (Name   => Null_Unbounded_String,
      N_Dims => 2,
      Dims   => [N, 1, 0, 0],
      Kind   => LLM_GGUF.GGML_TYPE_Q8_0,
      Offset => 0);

   function Info4 return LLM_GGUF.Tensor_Info is
     (Name   => Null_Unbounded_String,
      N_Dims => 2,
      Dims   => [N, 1, 0, 0],
      Kind   => LLM_GGUF.GGML_TYPE_Q4_0,
      Offset => 0);

   --  Max round-trip error of S (already-quantized bytes) vs X via the engine.
   function Deq_Err (X : Tensor; S : String; I : LLM_GGUF.Tensor_Info)
                     return Float
   is
      Q : constant Tensor := LLM_Dequant.Dequantize (I, S);
      M : Float := 0.0;
   begin
      for K in 1 .. N loop
         if abs (Get_Flat (X, K) - Get_Flat (Q, K)) > M then
            M := abs (Get_Flat (X, K) - Get_Flat (Q, K));
         end if;
      end loop;
      return M;
   end Deq_Err;

   function RT_Err  (X : Tensor) return Float is
     (Deq_Err (X, LLM_Quant.Quantize_Q8_0 (X), Info));
   function RT_Err4 (X : Tensor) return Float is
     (Deq_Err (X, LLM_Quant.Quantize_Q4_0 (X), Info4));

begin
   Put_Line ("=== FP32 -> Q8_0 quantizer ===");

   --  Varied values in ~[-1, 1]; per-block scale ~ 1/127, so error <~ half-step.
   declare
      X : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop
         Set_Flat (X, I, Float (((I * 37) mod 200) - 100) / 110.0);
      end loop;
      Check ("varied values round-trip within Q8_0 bound", RT_Err (X) < 0.01);
   end;

   --  Constant block quantizes near-exactly (the max element maps to 127).
   declare
      X : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop Set_Flat (X, I, 0.5); end loop;
      Check ("constant block is near-exact", RT_Err (X) < 1.0e-3);
   end;

   --  All-zero block: scale 0, all codes 0, exact.
   declare
      X : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop Set_Flat (X, I, 0.0); end loop;
      Check ("all-zero block is exact", RT_Err (X) = 0.0);
   end;

   --  Larger magnitudes scale proportionally (error still ~ amax/254).
   declare
      X : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop
         Set_Flat (X, I, Float (((I * 37) mod 200) - 100) / 110.0 * 50.0);
      end loop;
      Check ("large magnitudes round-trip (~50x scale)", RT_Err (X) < 0.5);
   end;

   New_Line; Put_Line ("--- Q4_0 (4-bit) ---");

   --  Q4_0: 4-bit, so the bound is coarser (~ half of amax/8).
   declare
      X : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop
         Set_Flat (X, I, Float (((I * 37) mod 200) - 100) / 110.0);
      end loop;
      Check ("Q4_0 varied values within 4-bit bound", RT_Err4 (X) < 0.1);
   end;

   declare
      X : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop Set_Flat (X, I, 0.0); end loop;
      Check ("Q4_0 all-zero is exact", RT_Err4 (X) = 0.0);
   end;

   --  The max-abs element is exact under Q4_0's signed-scale convention.
   declare
      X : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop Set_Flat (X, I, 0.05); end loop;
      Set_Flat (X, 7, -0.9);   -- the dominant element
      declare
         S : constant String := LLM_Quant.Quantize_Q4_0 (X);
         Q : constant Tensor := LLM_Dequant.Dequantize (Info4, S);
      begin
         Check ("Q4_0 max-abs element reproduced exactly",
                abs (Get_Flat (Q, 7) - (-0.9)) < 1.0e-3);
      end;
   end;

   New_Line; Put_Line ("--- Q5_0 (legacy 5-bit) ---");
   declare
      Info50 : constant LLM_GGUF.Tensor_Info :=
        (Name => Null_Unbounded_String, N_Dims => 2, Dims => [N, 1, 0, 0],
         Kind => LLM_GGUF.GGML_TYPE_Q5_0, Offset => 0);
      function RT_Err50 (X : Tensor) return Float is
        (Deq_Err (X, LLM_Quant.Quantize_Q5_0 (X), Info50));
   begin
      --  5-bit symmetric: finer than Q4_0 (32 levels vs 16) but coarser than Q8.
      declare
         X : Tensor := New_Tensor ([1, N]);
      begin
         for I in 1 .. N loop
            Set_Flat (X, I, Float (((I * 37) mod 200) - 100) / 110.0);
         end loop;
         Check ("Q5_0 varied values within 5-bit bound", RT_Err50 (X) < 0.05);
      end;

      declare
         X : Tensor := New_Tensor ([1, N]);
      begin
         for I in 1 .. N loop Set_Flat (X, I, 0.0); end loop;
         Check ("Q5_0 all-zero is exact", RT_Err50 (X) = 0.0);
      end;

      --  The max-abs element is exact under Q5_0's signed-scale (d = vmax/-16).
      declare
         X : Tensor := New_Tensor ([1, N]);
      begin
         for I in 1 .. N loop Set_Flat (X, I, 0.05); end loop;
         Set_Flat (X, 7, -0.9);
         declare
            S : constant String := LLM_Quant.Quantize_Q5_0 (X);
            Q : constant Tensor := LLM_Dequant.Dequantize (Info50, S);
         begin
            Check ("Q5_0 max-abs element reproduced exactly",
                   abs (Get_Flat (Q, 7) - (-0.9)) < 1.0e-3);
         end;
      end;
   end;

   New_Line; Put_Line ("--- Q4_K (256-element super-blocks) ---");
   declare
      NK    : constant := 256;
      InfoK : constant LLM_GGUF.Tensor_Info :=
        (Name => Null_Unbounded_String, N_Dims => 2, Dims => [NK, 1, 0, 0],
         Kind => LLM_GGUF.GGML_TYPE_Q4_K, Offset => 0);
      X : Tensor := New_Tensor ([1, NK]);
      function ErrK return Float is
         S : constant String := LLM_Quant.Quantize_Q4_K (X);
         Q : constant Tensor := LLM_Dequant.Dequantize (InfoK, S);
         M : Float := 0.0;
      begin
         for I in 1 .. NK loop
            if abs (Get_Flat (X, I) - Get_Flat (Q, I)) > M then
               M := abs (Get_Flat (X, I) - Get_Flat (Q, I));
            end if;
         end loop;
         return M;
      end ErrK;
   begin
      for I in 1 .. NK loop
         Set_Flat (X, I, Float (((I * 37) mod 200) - 100) / 110.0);
      end loop;
      Check ("Q4_K varied values within 4-bit bound", ErrK < 0.15);

      for I in 1 .. NK loop Set_Flat (X, I, 0.0); end loop;
      Check ("Q4_K all-zero is exact", ErrK = 0.0);

      --  Mix an all-positive sub-block (0..31) with spanning sub-blocks: the
      --  shared-dmin sign handling must still round-trip.
      for I in 1 .. NK loop
         Set_Flat (X, I, (if I <= 32 then 0.6
                          else Float (((I * 13) mod 100) - 50) / 60.0));
      end loop;
      Check ("Q4_K mixed positive/spanning sub-blocks within bound",
             ErrK < 0.15);
   end;

   New_Line; Put_Line ("--- Q5_K (5-bit super-blocks) ---");
   declare
      NK    : constant := 256;
      Info5 : constant LLM_GGUF.Tensor_Info :=
        (Name => Null_Unbounded_String, N_Dims => 2, Dims => [NK, 1, 0, 0],
         Kind => LLM_GGUF.GGML_TYPE_Q5_K, Offset => 0);
      X : Tensor := New_Tensor ([1, NK]);
      function Err5 return Float is
         S : constant String := LLM_Quant.Quantize_Q5_K (X);
         Q : constant Tensor := LLM_Dequant.Dequantize (Info5, S);
         M : Float := 0.0;
      begin
         for I in 1 .. NK loop
            M := Float'Max (M, abs (Get_Flat (X, I) - Get_Flat (Q, I)));
         end loop;
         return M;
      end Err5;
   begin
      --  5-bit affine: tighter than Q4_K (32 levels per sub-block vs 16).
      for I in 1 .. NK loop
         Set_Flat (X, I, Float (((I * 37) mod 200) - 100) / 110.0);
      end loop;
      Check ("Q5_K varied values within 5-bit bound", Err5 < 0.05);

      for I in 1 .. NK loop Set_Flat (X, I, 0.0); end loop;
      Check ("Q5_K all-zero is exact", Err5 = 0.0);

      --  All-positive sub-block (exercises the shared neg-min sign path) mixed
      --  with spanning ones — same as the Q4_K stress case, must round-trip.
      for I in 1 .. NK loop
         Set_Flat (X, I, (if I <= 32 then 0.6
                          else Float (((I * 13) mod 100) - 50) / 60.0));
      end loop;
      Check ("Q5_K mixed positive/spanning sub-blocks within bound", Err5 < 0.05);
   end;

   New_Line; Put_Line ("--- Q6_K (6-bit super-blocks) ---");
   declare
      NK    : constant := 256;
      Info6 : constant LLM_GGUF.Tensor_Info :=
        (Name => Null_Unbounded_String, N_Dims => 2, Dims => [NK, 1, 0, 0],
         Kind => LLM_GGUF.GGML_TYPE_Q6_K, Offset => 0);
      X : Tensor := New_Tensor ([1, NK]);
      --  RELATIVE error (max abs error / max |x|): the right metric for Q6_K,
      --  whose shared f16 d makes the absolute error scale with magnitude. The
      --  6-bit grid + per-16 scale gives ~3% relative regardless of range.
      function Err6 return Float is
         S   : constant String := LLM_Quant.Quantize_Q6_K (X);
         Q   : constant Tensor := LLM_Dequant.Dequantize (Info6, S);
         M   : Float := 0.0;
         Rng : Float := 0.0;
      begin
         for I in 1 .. NK loop
            M   := Float'Max (M, abs (Get_Flat (X, I) - Get_Flat (Q, I)));
            Rng := Float'Max (Rng, abs Get_Flat (X, I));
         end loop;
         return (if Rng > 0.0 then M / Rng else M);
      end Err6;
   begin
      for I in 1 .. NK loop
         Set_Flat (X, I, Float (((I * 37) mod 200) - 100) / 110.0);
      end loop;
      Check ("Q6_K varied values within 6-bit bound", Err6 < 0.05);

      for I in 1 .. NK loop Set_Flat (X, I, 0.0); end loop;
      Check ("Q6_K all-zero is exact", Err6 = 0.0);

      --  16x dynamic range ACROSS groups: the shared d still holds each group
      --  to the same ~3% relative bound (the signed per-16 scale absorbs it).
      for I in 1 .. NK loop
         Set_Flat (X, I, Float (((I * 53) mod 100) - 50) / 50.0
                          * Float (1 + (I / 16)));
      end loop;
      Check ("Q6_K wide per-group dynamic range within bound", Err6 < 0.05);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Quant;
