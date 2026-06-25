------------------------------------------------------------------------
-- train_resident_probe — Step 5b: Ada drives GPU-RESIDENT training through the
-- Train_GPU_Resident FFI. On a CPU host (no libaspidatrain.so) it loads
-- gracefully and reports SKIP/PASS; on a GPU host (ASPIDA_TRAIN_LIB set) it
-- creates a resident session, uploads X and T=X·W* once, steps on the device,
-- and the loss must collapse — proving Ada trains on the GPU with no per-op
-- host round-trips.
------------------------------------------------------------------------

with Ada.Text_IO;          use Ada.Text_IO;
with Ada.Command_Line;     use Ada.Command_Line;
with Interfaces.C;         use type Interfaces.C.C_float;
with Train_GPU_Resident;   use Train_GPU_Resident;

procedure Train_Resident_Probe is
   L     : constant := 4;
   B     : constant := 64;
   D     : constant := 128;
   Steps : constant := 2000;

   X    : F32_Array (0 .. B * D - 1);
   T    : F32_Array (0 .. B * D - 1);
   Wst  : F32_Array (0 .. D * D - 1);

   Seed : Long_Long_Integer := 12345;
   function Rnd return C_Float is   -- LCG in [-0.5, 0.5]
   begin
      Seed := (Seed * 1103515245 + 12345) mod 2147483648;
      return C_Float (Float (Seed) / 2147483648.0 - 0.5);
   end Rnd;
begin
   Put_Line ("=== train_resident_probe: Ada drives GPU-resident training ===");
   if not Available then
      Put_Line ("RESULT: PASS (binding compiles, links, dlopens gracefully;"
                & " set ASPIDA_TRAIN_LIB to a GPU shim to train)");
      return;
   end if;

   --  ---- Step 5c: grad-check the GPU backward vs a CPU finite difference ----
   declare
      GL  : constant := 2;
      GB  : constant := 4;
      GD  : constant := 16;
      GX  : F32_Array (0 .. GB * GD - 1);
      GT  : F32_Array (0 .. GB * GD - 1);
      GW  : F32_Array (0 .. GD * GD - 1);
      Eps : constant Float := 1.0E-3;
      Max_Rel : Float := 0.0;
      Bad : Natural := 0;   -- weights failing BOTH rel and abs tolerance
   begin
      for I in GX'Range loop GX (I) := Rnd; end loop;
      for I in GW'Range loop GW (I) := Rnd * 0.2; end loop;
      for R in 0 .. GB - 1 loop
         for Col in 0 .. GD - 1 loop
            declare S : C_Float := 0.0;
            begin
               for K in 0 .. GD - 1 loop S := S + GX (R * GD + K) * GW (K * GD + Col); end loop;
               GT (R * GD + Col) := S;
            end;
         end loop;
      end loop;
      declare
         Sg : constant Session := Create (GL, GB, GD, 1.0E-3);
      begin
         Set_Data (Sg, GX, GT);
         Put_Line ("  grad-check (E = 0.5*sum (Y-T)^2):");
         for Trial in 0 .. 5 loop
            declare
               Layer : constant Integer := Trial mod GL;
               Idx   : constant Integer := (Trial * 37 + 5) mod (GD * GD);
               Ga    : constant Float := Grad_At (Sg, Layer, Idx);   -- GPU analytic
               W0    : constant Float := W_Get (Sg, Layer, Idx);
               Lp, Lm, Gfd, Rel, Abs_D : Float;
               Ok_I : Boolean;
            begin
               W_Set (Sg, Layer, Idx, W0 + Eps); Lp := Loss_Only (Sg);
               W_Set (Sg, Layer, Idx, W0 - Eps); Lm := Loss_Only (Sg);
               W_Set (Sg, Layer, Idx, W0);                            -- restore
               Gfd   := (Lp - Lm) / (2.0 * Eps);                      -- CPU finite diff
               Abs_D := abs (Ga - Gfd);
               Rel   := Abs_D / (abs (Ga) + abs (Gfd) + 1.0E-9);
               --  Pass on relative error OR (for near-zero grads, where FP32
               --  finite-diff cancellation dominates) a tight absolute floor.
               Ok_I  := Rel < 2.0E-2 or else Abs_D < 5.0E-5;
               if not Ok_I then Bad := Bad + 1; end if;
               Max_Rel := Float'Max (Max_Rel, Rel);
               Put_Line ("    L" & Layer'Image & " i" & Idx'Image
                         & "  analytic=" & Ga'Image & "  fd=" & Gfd'Image
                         & "  rel=" & Rel'Image & (if Ok_I then "  ok" else "  BAD"));
            end;
         end loop;
         Free (Sg);
      end;
      New_Line;
      if Bad = 0 then
         Put_Line ("RESULT(5c): PASS (GPU grad matches finite-diff on all probes;"
                   & " max rel=" & Max_Rel'Image & ")");
      else
         Put_Line ("RESULT(5c): FAIL (" & Bad'Image & " probe(s) off; max rel="
                   & Max_Rel'Image & ")");
         Set_Exit_Status (Failure);
      end if;
   end;
   New_Line;

   --  ---- Step 5b: full resident training run (loss must collapse) ----
   --  X random; T = X·W* for a random W* (a reachable linear target).
   for I in X'Range loop X (I) := Rnd; end loop;
   for I in Wst'Range loop Wst (I) := Rnd * 0.2; end loop;
   for R in 0 .. B - 1 loop
      for Col in 0 .. D - 1 loop
         declare S : C_Float := 0.0;
         begin
            for K in 0 .. D - 1 loop
               S := S + X (R * D + K) * Wst (K * D + Col);
            end loop;
            T (R * D + Col) := S;
         end;
      end loop;
   end loop;

   declare
      Sess : constant Session := Create (L, B, D, 2.0E-3);
      Loss : Float := 1.0;
   begin
      Set_Data (Sess, X, T);                       -- upload ONCE (resident)
      for I in 1 .. Steps loop
         Loss := Step (Sess);                      -- resident fwd+bwd+AdamW
         if I = 1 or else I mod 500 = 0 or else I = Steps then
            Put_Line ("  step" & I'Image & "  loss/elem=" & Loss'Image);
         end if;
      end loop;
      Free (Sess);

      New_Line;
      if Loss < 1.0E-3 then
         Put_Line ("RESULT(5b): PASS (Ada drove GPU-resident training; loss collapsed)");
      else
         Put_Line ("RESULT(5b): FAIL (loss did not collapse:" & Loss'Image & ")");
         Set_Exit_Status (Failure);
      end if;
   end;
end Train_Resident_Probe;
