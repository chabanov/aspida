------------------------------------------------------------------------
-- q4k_export_demo — exercises the from-scratch Q4_K writer end-to-end. The
-- tiny q8_export_demo model has ne0=32 (< 256), so it can only reach Q8_0/
-- Q4_0; Q4_K needs rows that are a multiple of 256. This trains a Dm=256
-- model, exports it as F32 and Q4_K, loads BOTH in the real inference engine
-- (LLM_Llama), and shows the Q4_K file is ~8x smaller yet serves the same
-- predictions — proving Quantize_Q4_K -> Add_Tensor_Q4_K -> engine round-trips.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Directories;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Train;                 use Train;
with Student;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Q4K_Export_Demo is
   Voc : constant := 8;
   --  Dm and Ff are multiples of 256 so every weight row (ne0) is a whole
   --  number of Q4_K super-blocks; otherwise Add_W falls back to F32.
   package S is new Student
     (Voc => Voc, Dm => 256, Ff => 256, Seq => 3, Lyr => 1, Heads => 4,
      Use_RoPE => True);

   --  Heap-allocate: a Dm=256 model holds many 256x256 weight/grad/Adam
   --  matrices (~10 MB), which would overflow the 8 MB main-task stack.
   type Model_Ptr is access S.Model;
   MP : constant Model_Ptr := new S.Model;
   M  : S.Model renames MP.all;
   G  : RNG := Seeded (3.0);
   function Rnd (N : Integer) return Integer is
     (Integer (Real'Floor (Uniform (G) * Real (N))));

   F32_Path : constant String := "/tmp/q4kdemo_f32.gguf";
   Q4K_Path : constant String := "/tmp/q4kdemo_q4k.gguf";
   Q5K_Path : constant String := "/tmp/q4kdemo_q5k.gguf";
   Q6K_Path : constant String := "/tmp/q4kdemo_q6k.gguf";
begin
   Put_Line ("=== train -> quantize (Q4_K / Q5_K / Q6_K) -> serve ===");

   --  Train: token t (at position 1) -> successor (t+1) mod Voc.
   S.Init (M, 3.0);
   declare
      Toks : Label_Array (1 .. 3); L : S.Logit_Mat;
      P    : Matrix (1 .. 3, 1 .. Voc); Tgt : S.Logit_Mat; Loss : Real := 0.0;
      pragma Unreferenced (Loss);
   begin
      for Step in 1 .. 2000 loop
         declare T : constant Integer := Rnd (Voc); begin
            Toks := [T, 0, 0];
            S.Forward (M, Toks, L); Softmax_Rows (L, P); Tgt := P;
            for C in 1 .. Voc loop Tgt (1, C) := 0.0; end loop;
            Tgt (1, (T + 1) mod Voc + 1) := 1.0;
            Loss := S.Backward (M, Tgt); S.Step (M, 3.0e-3, Clip => 1.0);
         end;
      end loop;
   end;

   declare
      Toks_S : GGUF_Write.Str_List (1 .. Voc);
   begin
      for D in 0 .. Voc - 1 loop
         Toks_S (D + 1) := To_Unbounded_String (Integer'Image (D) (2 .. 2));
      end loop;
      S.Export_GGUF (M, F32_Path, Toks_S, Ctx => 64, Fmt => S.Q_None);
      S.Export_GGUF (M, Q4K_Path, Toks_S, Ctx => 64, Fmt => S.Q_Q4_K);
      S.Export_GGUF (M, Q5K_Path, Toks_S, Ctx => 64, Fmt => S.Q_Q5_K);
      S.Export_GGUF (M, Q6K_Path, Toks_S, Ctx => 64, Fmt => S.Q_Q6_K);
   end;

   Put_Line ("  F32  GGUF:" & Ada.Directories.Size (F32_Path)'Image & " bytes");
   Put_Line ("  Q4_K GGUF:" & Ada.Directories.Size (Q4K_Path)'Image & " bytes");
   Put_Line ("  Q5_K GGUF:" & Ada.Directories.Size (Q5K_Path)'Image & " bytes");
   Put_Line ("  Q6_K GGUF:" & Ada.Directories.Size (Q6K_Path)'Image & " bytes");

   declare
      LM_F   : constant LLM_Llama.Llama_Model := LLM_Llama.Load (F32_Path);
      LM_Q4K : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Q4K_Path);
      LM_Q5K : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Q5K_Path);
      LM_Q6K : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Q6K_Path);

      function Pred (LM : LLM_Llama.Llama_Model; T : Integer) return Integer is
         Ids  : constant LLM_Tokenizer.Token_Array (1 .. 3) := [T, 0, 0];
         Fl   : constant LLM_Llama.Logits_Flat := LLM_Llama.Forward_Logits (LM, Ids);
         Best : Integer := 0; BV : Float := Float'First;
      begin
         for C in 0 .. Voc - 1 loop
            if Fl (C) > BV then BV := Fl (C); Best := C; end if;
         end loop;
         return Best;
      end Pred;

      Acc_F, Acc_Q4K, Acc_Q5K, Acc_Q6K : Integer := 0;
   begin
      for T in 0 .. Voc - 1 loop
         if Pred (LM_F,   T) = (T + 1) mod Voc then Acc_F   := Acc_F   + 1; end if;
         if Pred (LM_Q4K, T) = (T + 1) mod Voc then Acc_Q4K := Acc_Q4K + 1; end if;
         if Pred (LM_Q5K, T) = (T + 1) mod Voc then Acc_Q5K := Acc_Q5K + 1; end if;
         if Pred (LM_Q6K, T) = (T + 1) mod Voc then Acc_Q6K := Acc_Q6K + 1; end if;
      end loop;
      Put_Line ("  served accuracy  F32:" & Acc_F'Image
                & "   Q4_K:" & Acc_Q4K'Image & "   Q5_K:" & Acc_Q5K'Image
                & "   Q6_K:" & Acc_Q6K'Image & "   (/" & Voc'Image & ")");
   end;
end Q4K_Export_Demo;
