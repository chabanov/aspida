------------------------------------------------------------------------
-- q8_export_demo — closes the train -> quantize -> serve loop. Trains a tiny
-- model (successor function), exports it as BOTH F32 and Q8_0 GGUFs, loads
-- both in the real inference engine (LLM_Llama), and shows the Q8_0 file is
-- ~4x smaller yet serves the same predictions.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Directories;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Train;                 use Train;
with Student;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Q8_Export_Demo is
   Voc : constant := 8;
   package S is new Student
     (Voc => Voc, Dm => 32, Ff => 64, Seq => 3, Lyr => 1, Heads => 2,
      Use_RoPE => True);

   M : S.Model;
   G : RNG := Seeded (3.0);
   function Rnd (N : Integer) return Integer is
     (Integer (Real'Floor (Uniform (G) * Real (N))));

   F32_Path : constant String := "/tmp/q8demo_f32.gguf";
   Q8_Path  : constant String := "/tmp/q8demo_q8.gguf";
   Q4_Path  : constant String := "/tmp/q8demo_q4.gguf";
begin
   Put_Line ("=== train -> quantize (Q8_0) -> serve ===");

   --  Train: token t (at position 1) -> successor (t+1) mod Voc.
   S.Init (M, 3.0);
   declare
      Toks : Label_Array (1 .. 3); L : S.Logit_Mat;
      P    : Matrix (1 .. 3, 1 .. Voc); Tgt : S.Logit_Mat; Loss : Real := 0.0;
      pragma Unreferenced (Loss);
   begin
      for Step in 1 .. 12000 loop
         declare T : constant Integer := Rnd (Voc); begin
            Toks := [T, 0, 0];
            S.Forward (M, Toks, L); Softmax_Rows (L, P); Tgt := P;
            for C in 1 .. Voc loop Tgt (1, C) := 0.0; end loop;
            Tgt (1, (T + 1) mod Voc + 1) := 1.0;
            Loss := S.Backward (M, Tgt); S.Step (M, 5.0e-3, Clip => 1.0);
         end;
      end loop;
   end;

   --  Export the same trained weights as F32 and as Q8_0.
   declare
      Toks_S : GGUF_Write.Str_List (1 .. Voc);
   begin
      for D in 0 .. Voc - 1 loop
         Toks_S (D + 1) := To_Unbounded_String (Integer'Image (D) (2 .. 2));
      end loop;
      S.Export_GGUF (M, F32_Path, Toks_S, Ctx => 64, Fmt => S.Q_None);
      S.Export_GGUF (M, Q8_Path,  Toks_S, Ctx => 64, Fmt => S.Q_Q8_0);
      S.Export_GGUF (M, Q4_Path,  Toks_S, Ctx => 64, Fmt => S.Q_Q4_0);
   end;

   Put_Line ("  F32  GGUF:" & Ada.Directories.Size (F32_Path)'Image & " bytes");
   Put_Line ("  Q8_0 GGUF:" & Ada.Directories.Size (Q8_Path)'Image & " bytes");
   Put_Line ("  Q4_0 GGUF:" & Ada.Directories.Size (Q4_Path)'Image & " bytes");

   --  Load all three with the real engine and compare predictions.
   declare
      LM_F  : constant LLM_Llama.Llama_Model := LLM_Llama.Load (F32_Path);
      LM_Q8 : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Q8_Path);
      LM_Q4 : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Q4_Path);

      function Pred (LM : LLM_Llama.Llama_Model; T : Integer) return Integer is
         Ids  : constant LLM_Tokenizer.Token_Array (1 .. 3) := [T, 0, 0];
         Fl   : constant LLM_Llama.Logits_Flat := LLM_Llama.Forward_Logits (LM, Ids);
         Best : Integer := 0; BV : Float := Float'First;
      begin
         for C in 0 .. Voc - 1 loop      -- row 0 predicts the next token
            if Fl (C) > BV then BV := Fl (C); Best := C; end if;
         end loop;
         return Best;
      end Pred;

      Acc_F, Acc_Q8, Acc_Q4 : Integer := 0;
   begin
      for T in 0 .. Voc - 1 loop
         if Pred (LM_F,  T) = (T + 1) mod Voc then Acc_F  := Acc_F  + 1; end if;
         if Pred (LM_Q8, T) = (T + 1) mod Voc then Acc_Q8 := Acc_Q8 + 1; end if;
         if Pred (LM_Q4, T) = (T + 1) mod Voc then Acc_Q4 := Acc_Q4 + 1; end if;
      end loop;
      Put_Line ("  served accuracy  F32:" & Acc_F'Image
                & "   Q8_0:" & Acc_Q8'Image & "   Q4_0:" & Acc_Q4'Image
                & "   (/" & Voc'Image & ")");
   end;
end Q8_Export_Demo;
