------------------------------------------------------------------------
-- cyber_train — prove the teach -> train -> serve loop on a REAL domain.
--
-- Learns a BPE vocabulary on a defensive-cybersecurity corpus, trains a small
-- from-scratch Student (next-token cross-entropy, RoPE / Llama-compatible),
-- exports it to GGUF, then RELOADS that GGUF with the real inference engine
-- (LLM_Llama) and (1) verifies the engine-served model reproduces the trained
-- model's next-token predictions exactly, and (2) greedily continues a few
-- cybersecurity prompts. Demonstration of the pipeline on real data — the
-- model is intentionally tiny, so this proves the loop, not capability.
--
--   ./obj/cyber_train [corpus.txt]      (default: tools/cyber_corpus.txt)
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Command_Line;      use Ada.Command_Line;
with Ada.Streams.Stream_IO;
with Train;       use Train;
with Student;
with BPE_Train;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Cyber_Train is
   Voc    : constant := 512;     -- fixed vocab (BPE trained to this target)
   Dm     : constant := 64;
   Ff     : constant := 128;
   Seq    : constant := 32;      -- training/serving context window
   Lyr    : constant := 2;
   Heads  : constant := 4;
   Epochs : constant := 25;
   Base_LR : constant := 4.0E-3;

   package S is new Student
     (Voc => Voc, Dm => Dm, Ff => Ff, Seq => Seq, Lyr => Lyr, Heads => Heads,
      Use_RoPE => True, Rope_Base => 10000.0);
   type Model_Acc is access S.Model;
   M : constant Model_Acc := new S.Model;

   --------------------------------------------------------------------
   function Slurp (Path : String) return String is
      package SIO renames Ada.Streams.Stream_IO;
      F : SIO.File_Type;
   begin
      SIO.Open (F, SIO.In_File, Path);
      declare
         Len : constant Natural := Natural (SIO.Size (F));
         Buf : String (1 .. Len);
      begin
         if Len > 0 then
            String'Read (SIO.Stream (F), Buf);
         end if;
         SIO.Close (F);
         return Buf;
      end;
   end Slurp;

   Corpus_Path : constant String :=
     (if Ada.Command_Line.Argument_Count >= 1
      then Ada.Command_Line.Argument (1) else "tools/cyber_corpus.txt");

   Tk  : BPE_Train.Trainer;
begin
   Put_Line ("=== cyber_train: teach -> train -> serve on a cyber corpus ===");
   declare
      Corpus : constant String := Slurp (Corpus_Path);
   begin
      Put_Line ("corpus:" & Corpus'Length'Image & " bytes from " & Corpus_Path);
      BPE_Train.Train (Tk, Corpus, Target_Vocab => Voc);
      Put_Line ("BPE vocab:" & BPE_Train.Vocab_Size (Tk)'Image
                & "   (target" & Voc'Image & ")");

      declare
         VS  : constant Natural := BPE_Train.Vocab_Size (Tk);
         Ids : constant BPE_Train.Id_Array := BPE_Train.Encode (Tk, Corpus);
         NT  : constant Natural := Ids'Length;
      begin
         Put_Line ("corpus tokens:" & NT'Image);
         Flush;
         if NT <= Seq then
            Put_Line ("corpus too short for one window; aborting");
            return;
         end if;

         --  ---- train (next-token cross-entropy over sliding windows) ----
         S.Init (M.all, 7.0);
         declare
            Toks : Label_Array (1 .. Seq);
            L    : S.Logit_Mat;
            Tgt  : S.Logit_Mat;
            First_Loss, Last_Loss : Real := 0.0;
         begin
            for Ep in 1 .. Epochs loop
               declare
                  Sum_L : Real := 0.0;
                  N_Win : Natural := 0;
                  LR : constant Real :=
                    Base_LR * (1.0 - 0.85 * Real (Ep - 1) / Real (Epochs));
               begin
                  for St in Ids'First .. NT - Seq loop
                     for I in 1 .. Seq loop
                        Toks (I) := Ids (St + I - 1);          -- 0-based id
                     end loop;
                     S.Forward (M.all, Toks, L);
                     Tgt := [others => [others => 0.0]];
                     for P in 1 .. Seq loop
                        Tgt (P, Ids (St + P) + 1) := 1.0;      -- next token
                     end loop;
                     Sum_L := Sum_L + S.Backward (M.all, Tgt);
                     S.Step (M.all, LR, Clip => 1.0);
                     N_Win := N_Win + 1;
                  end loop;
                  if N_Win > 0 then
                     declare Avg : constant Real := Sum_L / Real (N_Win);
                     begin
                        if Ep = 1 then First_Loss := Avg; end if;
                        if Ep = Epochs then Last_Loss := Avg; end if;
                        if Ep = 1 or else Ep mod 5 = 0 or else Ep = Epochs then
                           Put_Line ("  epoch" & Ep'Image & "  avg loss="
                                     & Avg'Image);
                           Flush;   -- stream progress (stdout is block-buffered)
                        end if;
                     end;
                  end if;
               end;
            end loop;
            Put_Line ("loss: first epoch=" & First_Loss'Image
                      & "   final=" & Last_Loss'Image);
         end;

         --  ---- export to GGUF ----
         declare
            Toks_S : GGUF_Write.Str_List (1 .. Voc);
         begin
            for I in 1 .. Voc loop
               if I - 1 < VS then
                  Toks_S (I) := To_Unbounded_String (BPE_Train.Token_Piece (Tk, I - 1));
               else
                  Toks_S (I) := To_Unbounded_String ("<unused" & Integer'Image (I) & ">");
               end if;
            end loop;
            S.Export_GGUF (M.all, "cyber.gguf", Toks_S, Ctx => Seq);
            Put_Line ("exported cyber.gguf (" & VS'Image & " vocab)");
         end;

         --  ---- reload with the real engine and verify the loop ----
         declare
            LM : constant LLM_Llama.Llama_Model := LLM_Llama.Load ("cyber.gguf");
            Vc : constant Integer := LLM_Llama.Vocab_Size (LM);

            function Engine_Next (Ids_In : LLM_Tokenizer.Token_Array) return Integer is
               F   : constant LLM_Llama.Logits_Flat :=
                 LLM_Llama.Forward_Logits (LM, Ids_In);
               Row : constant Integer := (Ids_In'Length - 1) * Vc;  -- last pos
               Best : Integer := 0;
               BV   : Float := Float'First;
            begin
               --  Restrict to trained ids (< VS); unused rows are random.
               for D in 0 .. VS - 1 loop
                  if F (Row + D) > BV then BV := F (Row + D); Best := D; end if;
               end loop;
               return Best;
            end Engine_Next;

            function Student_Next (Ids_In : LLM_Tokenizer.Token_Array) return Integer is
               Toks : Label_Array (1 .. Seq) := [others => 0];
               L    : S.Logit_Mat;
               Best : Integer := 0;
               BV   : Real := Real'First;
            begin
               for I in Ids_In'Range loop Toks (I) := Ids_In (I); end loop;
               S.Forward (M.all, Toks, L);
               for D in 1 .. VS loop
                  if L (Ids_In'Length, D) > BV then BV := L (Ids_In'Length, D); Best := D - 1; end if;
               end loop;
               return Best;
            end Student_Next;

            --  Greedy continuation via the SERVED model, decoded with the BPE.
            function Continue (Prompt : String; N_New : Integer) return String is
               P0  : constant BPE_Train.Id_Array := BPE_Train.Encode (Tk, Prompt);
               Buf : LLM_Tokenizer.Token_Array (1 .. Seq) := [others => 0];
               Len : Integer := Integer'Min (P0'Length, Seq);
            begin
               for I in 1 .. Len loop Buf (I) := P0 (P0'First + I - 1); end loop;
               for K in 1 .. N_New loop
                  exit when Len >= Seq;
                  Len := Len + 1;
                  Buf (Len) := Engine_Next (Buf (1 .. Len - 1));
               end loop;
               declare
                  Out_Ids : BPE_Train.Id_Array (1 .. Len);
               begin
                  for I in 1 .. Len loop Out_Ids (I) := Natural (Buf (I)); end loop;
                  return BPE_Train.Decode (Tk, Out_Ids);
               end;
            end Continue;

            Match : Integer := 0;
            Trials : Integer := 0;
         begin
            Put_Line ("engine vocab:" & Vc'Image);

            --  (1) loop fidelity: served next-token == trained next-token at a
            --  range of corpus positions.
            declare
               P : constant Integer := Integer'Min (16, Seq - 1);
            begin
               for St in Ids'First .. Integer'Min (NT - P - 1, Ids'First + 60) loop
                  declare
                     W : LLM_Tokenizer.Token_Array (1 .. P);
                  begin
                     for I in 1 .. P loop W (I) := Ids (St + I - 1); end loop;
                     Trials := Trials + 1;
                     if Engine_Next (W) = Student_Next (W) then
                        Match := Match + 1;
                     end if;
                  end;
               end loop;
            end;
            Put_Line ("served==trained next-token:" & Match'Image & " /"
                      & Trials'Image
                      & (if Match = Trials then "   (loop is bit-faithful)" else ""));

            --  (2) qualitative: greedy continuations of cyber prompts.
            New_Line;
            Put_Line ("engine-served continuations:");
            Put_Line ("  [" & Continue ("A buffer overflow occurs when ", 14) & "]");
            Put_Line ("  [" & Continue ("To prevent SQL injection, ", 14) & "]");
            Put_Line ("  [" & Continue ("The principle of least privilege ", 14) & "]");

            New_Line;
            if Match = Trials and then Trials > 0 then
               Put_Line ("RESULT: PASS  (cyber model trained, exported, served; loop closed)");
            else
               Put_Line ("RESULT: FAIL  (served/trained mismatch — see counts)");
               Set_Exit_Status (Failure);
            end if;
         end;
      end;
   end;
end Cyber_Train;
