------------------------------------------------------------------------
-- train_aspida — train ASPIDA (2-digit addition) in a LLAMA-COMPATIBLE form
-- (RoPE instead of learned positions), save her, export to GGUF, then LOAD
-- that GGUF with the real inference engine (LLM_Llama) and verify she still
-- adds correctly — closing teach -> train -> serve for Aspida.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Train;       use Train;
with Student;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Train_Aspida is
   Voc : constant := 12;
   Dm  : constant := 128;
   Ff  : constant := 256;
   Seq : constant := 9;
   Lyr : constant := 3;
   Heads : constant := 4;
   Steps  : constant := 300_000;
   Warmup : constant := 2_000;
   Base_LR : constant := 2.0E-3;

   package S is new Student
     (Voc => Voc, Dm => Dm, Ff => Ff, Seq => Seq, Lyr => Lyr, Heads => Heads,
      Use_RoPE => True, Rope_Base => 10000.0);
   type Model_Acc is access S.Model;
   M : constant Model_Acc := new S.Model;
   G : RNG := Seeded (2025.0);

   function Rnd (N : Integer) return Integer is
     (Integer (Real'Floor (Uniform (G) * Real (N))));

   procedure Encode (A, B : Integer; Toks : out Label_Array) is
      Sum : constant Integer := A + B;
   begin
      Toks (1) := A / 10;  Toks (2) := A mod 10;  Toks (3) := 10;
      Toks (4) := B / 10;  Toks (5) := B mod 10;  Toks (6) := 11;
      Toks (7) := Sum mod 10;  Toks (8) := (Sum / 10) mod 10;  Toks (9) := (Sum / 100) mod 10;
   end Encode;

   function Argmax (L : S.Logit_Mat; Row : Integer) return Integer is
      Best : Integer := 0;  BV : Real := Real'First;
   begin
      for D in 0 .. 9 loop
         if L (Row, D + 1) > BV then BV := L (Row, D + 1); Best := D; end if;
      end loop;
      return Best;
   end Argmax;

   function Solve (A, B : Integer) return Integer is
      Toks : Label_Array (1 .. Seq); L : S.Logit_Mat; R1, R2, R3 : Integer;
   begin
      Toks := [A / 10, A mod 10, 10, B / 10, B mod 10, 11, 0, 0, 0];
      S.Forward (M.all, Toks, L); R1 := Argmax (L, 6); Toks (7) := R1;
      S.Forward (M.all, Toks, L); R2 := Argmax (L, 7); Toks (8) := R2;
      S.Forward (M.all, Toks, L); R3 := Argmax (L, 8);
      return R1 + 10 * R2 + 100 * R3;
   end Solve;

   function Accuracy (N : Integer) return Integer is
      Ok : Integer := 0;
   begin
      for I in 1 .. N loop
         declare A : constant Integer := Rnd (100); B : constant Integer := Rnd (100);
         begin if Solve (A, B) = A + B then Ok := Ok + 1; end if; end;
      end loop;
      return Ok;
   end Accuracy;

   Toks : Label_Array (1 .. Seq);
   L    : S.Logit_Mat;
   P    : Matrix (1 .. Seq, 1 .. Voc);
   Tgt  : S.Logit_Mat;
   Loss, LR : Real := 0.0;
begin
   Put_Line ("=== Training ASPIDA (2-digit add, RoPE / Llama-compatible) ===");
   S.Init (M.all, 2025.0);
   for Step in 1 .. Steps loop
      LR := Base_LR * Real'Min (1.0, Real (Step) / Real (Warmup))
                    * (1.0 - 0.9 * Real'Max (0.0, Real (Step - Warmup)) / Real (Steps - Warmup));
      declare A : constant Integer := Rnd (100); B : constant Integer := Rnd (100);
      begin
         Encode (A, B, Toks);
         S.Forward (M.all, Toks, L);
         Softmax_Rows (L, P);
         Tgt := P;
         for R in 6 .. 8 loop
            for C in 1 .. Voc loop Tgt (R, C) := 0.0; end loop;
            Tgt (R, Toks (R + 1) + 1) := 1.0;
         end loop;
         Loss := S.Backward (M.all, Tgt);
         S.Step (M.all, LR, Clip => 1.0);
      end;
      if Step mod 20000 = 0 then
         Put_Line ("  step" & Step'Image & "   loss=" & Loss'Image
                   & "   acc=" & Integer'Image (Accuracy (200)) & "/200");
      end if;
   end loop;
   Put_Line ("trained acc:" & Integer'Image (Accuracy (1000)) & "/1000");

   S.Save (M.all, "aspida.model");
   declare
      Toks_S : GGUF_Write.Str_List (1 .. Voc);
   begin
      for D in 0 .. 9 loop Toks_S (D + 1) := To_Unbounded_String (Integer'Image (D) (2 .. 2)); end loop;
      Toks_S (11) := To_Unbounded_String ("+");
      Toks_S (12) := To_Unbounded_String ("=");
      S.Export_GGUF (M.all, "aspida.gguf", Toks_S, Bos => 11, Eos => 11, Ctx => 64);
      Put_Line ("saved aspida.model + exported aspida.gguf");
   end;

   --  ---- load the GGUF with the real engine and verify ----
   declare
      LM : constant LLM_Llama.Llama_Model := LLM_Llama.Load ("aspida.gguf");
      Vc : constant Integer := LLM_Llama.Vocab_Size (LM);
      function Amax (F : LLM_Llama.Logits_Flat; Row : Integer) return Integer is
         Best : Integer := 0; BV : Float := Float'First;
      begin
         for D in 0 .. 9 loop
            if F (Row * Vc + D) > BV then BV := F (Row * Vc + D); Best := D; end if;
         end loop;
         return Best;
      end Amax;
      function Solve_E (A, B : Integer) return Integer is
         Ids : LLM_Tokenizer.Token_Array (1 .. Seq) :=
           [A / 10, A mod 10, 10, B / 10, B mod 10, 11, 0, 0, 0];
         R1, R2, R3 : Integer;
      begin
         R1 := Amax (LLM_Llama.Forward_Logits (LM, Ids), 5); Ids (7) := R1;
         R2 := Amax (LLM_Llama.Forward_Logits (LM, Ids), 6); Ids (8) := R2;
         R3 := Amax (LLM_Llama.Forward_Logits (LM, Ids), 7);
         return R1 + 10 * R2 + 100 * R3;
      end Solve_E;
      Ok : Integer := 0;
   begin
      for I in 1 .. 200 loop
         declare A : constant Integer := Rnd (100); B : constant Integer := Rnd (100);
         begin if Solve_E (A, B) = A + B then Ok := Ok + 1; end if; end;
      end loop;
      Put_Line ("ENGINE-served Aspida (from aspida.gguf):"
                & Integer'Image (Ok) & "/200 correct");
      for I in 1 .. 6 loop
         declare A : constant Integer := Rnd (100); B : constant Integer := Rnd (100);
                 Ans : constant Integer := Solve_E (A, B);
         begin
            Put_Line ("   engine:" & Integer'Image (A) & " +" & Integer'Image (B)
                      & " =" & Integer'Image (Ans)
                      & (if Ans = A + B then "  ok" else "  (should be" & Integer'Image (A + B) & ")"));
         end;
      end loop;
   end;
end Train_Aspida;
