------------------------------------------------------------------------
-- student_gpu_probe — Stream B: Ada drives full multi-head transformer training
-- on the GPU through Student_GPU, with a RUNTIME-CONFIGURABLE architecture (one
-- libaspidastudent.so serves any tier). Trains TWO differently-sized configs to
-- prove the shim is parameterised, not hardcoded. Graceful SKIP on a CPU host.
------------------------------------------------------------------------

with Ada.Text_IO;      use Ada.Text_IO;
with Ada.Command_Line; use Ada.Command_Line;
with Student_GPU;      use Student_GPU;

procedure Student_GPU_Probe is
   All_Ok : Boolean := True;

   procedure Run (Tag : String; Voc, Dim, Ff, Seq, Layers, Heads : Positive;
                  Ids, Tgts : Int_Array) is
      Steps : constant := 400;
      Sess  : constant Session := Create (Voc, Dim, Ff, Seq, Layers, Heads);
      Loss  : Float := 1.0;
   begin
      Set_Data (Sess, Ids, Tgts);
      for I in 1 .. Steps loop
         Loss := Step (Sess, 0.005);   --  AdamW lr
      end loop;
      Free (Sess);
      Put_Line ("  " & Tag & " (D=" & Dim'Image & " H=" & Heads'Image
                & " L=" & Layers'Image & " S=" & Seq'Image & "): final loss="
                & Loss'Image & (if Loss < 0.5 then "  ok" else "  HIGH"));
      if Loss >= 0.5 then All_Ok := False; end if;
   end Run;
begin
   Put_Line ("=== student_gpu_probe: Ada drives runtime-configurable GPU transformer ===");
   if not Available then
      Put_Line ("RESULT: PASS (binding compiles, links, dlopens gracefully;"
                & " set ASPIDA_STUDENT_LIB to the GPU shim to train)");
      return;
   end if;

   Run ("config A", 32, 16, 32, 6, 2, 2,
        [3, 7, 3, 11, 20, 7], [5, 1, 9, 0, 14, 2]);
   Run ("config B", 48, 24, 48, 8, 3, 3,
        [3, 7, 3, 11, 20, 7, 40, 2], [5, 1, 9, 0, 14, 2, 30, 8]);

   --  16-aligned config (S=16, dims multiples of 16): exercises the tensor-core
   --  (WMMA FP16) forward path; must still train.
   Run ("config WMMA(S16,aligned)", 32, 16, 32, 16, 2, 2,
        [3, 7, 3, 11, 20, 7, 1, 9, 4, 15, 8, 2, 19, 6, 12, 0],
        [5, 1, 9, 0, 14, 2, 7, 3, 11, 6, 1, 8, 4, 9, 2, 5]);

   --  gradient accumulation: G micro-batches per AdamW update (the Step-7 enabler)
   declare
      G    : constant := 4;
      Sess : constant Session := Create (32, 16, 32, 6, 2, 2);
      Loss : Float := 1.0;
      Ids  : constant Int_Array := [3, 7, 3, 11, 20, 7];
      Tgts : constant Int_Array := [5, 1, 9, 0, 14, 2];
   begin
      Set_Data (Sess, Ids, Tgts);
      for Macro in 1 .. 150 loop
         for Mi in 1 .. G loop
            Loss := Micro (Sess);          -- accumulate (one all-reduce would sum these across nodes)
         end loop;
         Apply (Sess, 0.005, G);           -- average over G + AdamW update
      end loop;
      Free (Sess);
      Put_Line ("  grad-accum (G=4 micro/update): final loss=" & Loss'Image
                & (if Loss < 0.5 then "  ok" else "  HIGH"));
      if Loss >= 0.5 then All_Ok := False; end if;
   end;

   New_Line;
   if All_Ok then
      Put_Line ("RESULT: PASS (Ada trained two runtime-configured transformers on GPU)");
   else
      Put_Line ("RESULT: FAIL");
      Set_Exit_Status (Failure);
   end if;
end Student_GPU_Probe;
