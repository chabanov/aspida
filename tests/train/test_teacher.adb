------------------------------------------------------------------------
-- test_teacher — the "teach" half with a REAL model: build a tiny student,
-- export it as a llama GGUF, load it through the actual inference engine, and
-- use it as a Distill teacher to capture a top-K distillation dataset. Proves
-- an existing model (run by our engine) can teach a new one — and cross-checks
-- the captured top-K against the engine's raw logits.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Train;
with GGUF_Write;
with Export_Llama;
with LLM_Llama;
with LLM_Tokenizer;
with Distill;
with Teacher_Llama;

procedure Test_Teacher is
   use type Distill.Logit;
   use type Distill.Token;
   Dim : constant := 16;
   F   : constant := 32;
   Vsz : constant := 24;
   K   : constant := 5;
   Path : constant String := "/tmp/aspida_teacher.gguf";
   Pass : Boolean := True;
   G : Train.RNG := Train.Seeded (9.0);

   E  : Train.Matrix (1 .. Vsz, 1 .. Dim);
   G1, G2, Gf : Train.Matrix (1 .. 1, 1 .. Dim);
   Wq, Wk, Wv, Wo : Train.Matrix (1 .. Dim, 1 .. Dim);
   Wg, Wu : Train.Matrix (1 .. Dim, 1 .. F);
   Wd : Train.Matrix (1 .. F, 1 .. Dim);
   Wout : Train.Matrix (1 .. Dim, 1 .. Vsz);
   procedure IW (W : out Train.Matrix) is begin Train.Init_Glorot (W, G); end IW;

   procedure Chk (Cond : Boolean; Name : String) is
   begin
      Put_Line ("  " & Name & ": " & (if Cond then "OK" else "FAIL"));
      if not Cond then Pass := False; end if;
   end Chk;
begin
   Put_Line ("=== Aspida teacher: real engine -> distillation dataset ===");
   IW (E); IW (Wq); IW (Wk); IW (Wv); IW (Wo); IW (Wg); IW (Wu); IW (Wd); IW (Wout);
   G1 := [others => [others => 1.0]];
   G2 := [others => [others => 1.0]];
   Gf := [others => [others => 1.0]];

   declare
      Toks : GGUF_Write.Str_List (1 .. Vsz);
   begin
      for I in 1 .. Vsz loop
         Toks (I) := To_Unbounded_String ("t" & Integer'Image (I - 1));
      end loop;
      Export_Llama.Save (Path, E, G1, G2, Gf, Wq, Wk, Wv, Wo, Wg, Wu, Wd, Wout, Toks);
      Put_Line ("  exported teacher model -> " & Path);
   end;

   declare
      M       : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Path);
      Teacher : Teacher_Llama.LM_Teacher := Teacher_Llama.Make (M);
      Prompt  : constant Distill.Token_Array := [1, 5, 7, 3, 9];
      S       : constant Distill.Sample := Distill.Capture (Teacher, Prompt, K);
   begin
      Put_Line ("  captured sample: N =" & S.N'Image & "  K =" & S.K'Image);
      Chk (S.N = Prompt'Length, "sample length matches prompt");

      --  top-K rows are descending and finite
      declare
         Ok_Desc, Ok_Fin : Boolean := True;
      begin
         for R in 1 .. S.N loop
            for J in 1 .. K loop
               if abs (Float (S.Top_Logit (R, J))) > 1.0E30 then Ok_Fin := False; end if;
            end loop;
            for J in 1 .. K - 1 loop
               if S.Top_Logit (R, J) < S.Top_Logit (R, J + 1) then Ok_Desc := False; end if;
            end loop;
         end loop;
         Chk (Ok_Fin, "all captured logits finite");
         Chk (Ok_Desc, "top-K descending per position");
      end;

      --  cross-check row 1 argmax against the engine's raw logits
      declare
         Ids  : constant LLM_Tokenizer.Token_Array := [1, 5, 7, 3, 9];
         Flat : constant LLM_Llama.Logits_Flat := LLM_Llama.Forward_Logits (M, Ids);
         Vc   : constant Integer := LLM_Llama.Vocab_Size (M);
         Best : Integer := 0;
         BestV : Float := Float'First;
      begin
         for C in 0 .. Vc - 1 loop
            if Flat (C) > BestV then BestV := Flat (C); Best := C; end if;
         end loop;
         Chk (Integer (S.Top_Ids (1, 1)) = Best,
              "captured argmax matches engine logits (id" & Best'Image & ")");
      end;

      --  dataset round-trip
      declare
         D : Distill.Dataset;
      begin
         D.Append (S);
         Distill.Write ("/tmp/aspida_teacher_ds.bin", D);
         declare
            D2 : constant Distill.Dataset := Distill.Read ("/tmp/aspida_teacher_ds.bin");
            Same : Boolean := Natural (D2.Length) = 1;
         begin
            if Same then
               for R in 1 .. S.N loop
                  for J in 1 .. K loop
                     if D2 (1).Top_Ids (R, J) /= S.Top_Ids (R, J)
                       or else D2 (1).Top_Logit (R, J) /= S.Top_Logit (R, J)
                     then Same := False; end if;
                  end loop;
               end loop;
            end if;
            Chk (Same, "distillation dataset round-trips");
         end;
      end;
   end;

   New_Line;
   if Pass then
      Put_Line ("RESULT: PASS  (existing model produced a distillation dataset)");
   else
      Put_Line ("RESULT: FAIL");
   end if;
end Test_Teacher;
