------------------------------------------------------------------------
-- test_train_gpu — exercises the Train_GPU FFI binding. On a CPU host the
-- shim is absent: the binding must load gracefully (PASS). On the GPU box
-- (ASPIDA_TRAIN_GPU=1 + libaspidatrain.so) it runs a real MM_Fwd on the GPU
-- and checks it against a CPU FP32 reference — proving the Ada->CUDA path.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train_GPU;   use Train_GPU;

procedure Test_Train_GPU is
   M : constant := 8;
   K : constant := 6;
   N : constant := 4;
   A   : F32_Array (0 .. M * K - 1);
   B   : F32_Array (0 .. K * N - 1);
   C   : F32_Array (0 .. M * N - 1);   -- filled by MM_Fwd (GPU)
   Ref : F32_Array (0 .. M * N - 1);
begin
   Put_Line ("=== Train_GPU FFI binding ===");
   if not Available then
      Put_Line ("  shim not loaded on this host (expected without GPU)");
      Put_Line ("RESULT: PASS (binding compiles, links, loads gracefully)");
      return;
   end if;

   for I in A'Range loop A (I) := C_Float (Float (I mod 7) * 0.1 - 0.3); end loop;
   for I in B'Range loop B (I) := C_Float (Float (I mod 5) * 0.1 - 0.2); end loop;
   MM_Fwd (A, B, C, M, K, N);

   for R in 0 .. M - 1 loop
      for Cc in 0 .. N - 1 loop
         declare
            S : Float := 0.0;
         begin
            for Kk in 0 .. K - 1 loop
               S := S + Float (A (R * K + Kk)) * Float (B (Kk * N + Cc));
            end loop;
            Ref (R * N + Cc) := C_Float (S);
         end;
      end loop;
   end loop;

   declare
      Maxd : Float := 0.0;
   begin
      for I in C'Range loop
         Maxd := Float'Max (Maxd, abs (Float (C (I)) - Float (Ref (I))));
      end loop;
      Put_Line ("  Ada->GPU MM_Fwd max abs diff =" & Maxd'Image);
      if Maxd < 1.0E-3 then
         Put_Line ("RESULT: PASS (Ada -> CUDA == CPU)");
      else
         Put_Line ("RESULT: FAIL");
      end if;
   end;
end Test_Train_GPU;
