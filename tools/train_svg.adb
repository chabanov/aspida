------------------------------------------------------------------------
-- train_svg — P4: train a byte-level Student to map an icon spec -> compact SVG
-- (verifier-filtered data from svg_icons.py), export a GGUF, load it in the real
-- engine, and GENERATE an SVG for each held-out prefix. The generated SVGs are
-- written to student_svgs.txt; svg_icons.py then renders + verifies them to get
-- the student pass-rate (vs the teacher baseline). Mirrors train_aspida.
--
--   reads:  <dir>/train.txt (prefix+svg per line), <dir>/eval.txt (prefix per line)
--   writes: <dir>/student_svgs.txt (one generated SVG per held-out prefix)
------------------------------------------------------------------------

with Ada.Text_IO;          use Ada.Text_IO;
with Ada.Command_Line;     use Ada.Command_Line;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Train;                use Train;
with Student;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Train_SVG is
   Voc : constant := 257;     -- 0..255 bytes + 256 = EOS/pad
   EOS : constant := 256;
   Dm  : constant := 64; Ff : constant := 128;
   Seq : constant := 120; Lyr : constant := 2; Heads : constant := 4;
   Steps : constant := 30_000; Warmup : constant := 800; Base_LR : constant := 1.5E-3;

   Dir : constant String := (if Argument_Count >= 1 then Argument (1) else "svgdata");

   package S is new Student
     (Voc => Voc, Dm => Dm, Ff => Ff, Seq => Seq, Lyr => Lyr, Heads => Heads,
      Use_RoPE => True, Rope_Base => 10000.0);
   type Model_Acc is access S.Model;
   M : constant Model_Acc := new S.Model;
   G : RNG := Seeded (2025.0);
   function Rnd (N : Integer) return Integer is (Integer (Real'Floor (Uniform (G) * Real (N))));

   Max_Ex : constant := 512;
   type Seq_Arr is array (1 .. Seq) of Integer;
   Train_X : array (1 .. Max_Ex) of Seq_Arr;
   Train_L : array (1 .. Max_Ex) of Natural;   -- real length (<= Seq), rest are EOS
   N_Ex : Natural := 0;

   --  Encode a line (string) into a padded byte sequence + EOS; return real length.
   procedure Encode (Line : String; Sq : out Seq_Arr; Len : out Natural) is
   begin
      Sq := [others => EOS];
      Len := 0;
      for I in Line'Range loop
         exit when Len >= Seq - 1;
         Len := Len + 1; Sq (Len) := Character'Pos (Line (I));
      end loop;
      Len := Len + 1; Sq (Len) := EOS;          -- terminator the student learns to emit
   end Encode;

   procedure Load_Train is
      F : File_Type;
   begin
      Open (F, In_File, Dir & "/train.txt");
      while not End_Of_File (F) and then N_Ex < Max_Ex loop
         declare L : constant String := Get_Line (F);
         begin
            if L'Length > 0 then
               N_Ex := N_Ex + 1; Encode (L, Train_X (N_Ex), Train_L (N_Ex));
            end if;
         end;
      end loop;
      Close (F);
   end Load_Train;

   Toks : Label_Array (1 .. Seq);
   L    : S.Logit_Mat;
   P    : Matrix (1 .. Seq, 1 .. Voc);
   Tgt  : S.Logit_Mat;
   Loss, LR : Real := 0.0;
begin
   Put_Line ("=== train_svg: byte-level spec->SVG student ===");
   Load_Train;
   Put_Line ("loaded" & N_Ex'Image & " verified training examples");
   S.Init (M.all, 2025.0);

   for Step in 1 .. Steps loop
      LR := Base_LR * Real'Min (1.0, Real (Step) / Real (Warmup))
                    * (1.0 - 0.9 * Real'Max (0.0, Real (Step - Warmup)) / Real (Steps - Warmup));
      declare
         E : constant Integer := 1 + Rnd (N_Ex);
      begin
         for I in 1 .. Seq loop Toks (I) := Train_X (E)(I); end loop;
         S.Forward (M.all, Toks, L);
         Softmax_Rows (L, P);
         Tgt := P;
         for R in 1 .. Seq - 1 loop                    -- next-token target on every position
            for C in 1 .. Voc loop Tgt (R, C) := 0.0; end loop;
            Tgt (R, Toks (R + 1) + 1) := 1.0;
         end loop;
         Loss := S.Backward (M.all, Tgt);
         S.Step (M.all, LR, Clip => 1.0);
      end;
      if Step mod 5000 = 0 then
         Put_Line ("  step" & Step'Image & "  loss=" & Loss'Image); Flush;
      end if;
   end loop;
   Put_Line ("trained, final loss=" & Loss'Image);

   --  export GGUF
   declare
      TS : GGUF_Write.Str_List (1 .. Voc);
      H  : constant String := "0123456789abcdef";
   begin
      for I in 0 .. 255 loop
         TS (I + 1) := To_Unbounded_String ([1 => H (I / 16 + 1), 2 => H (I mod 16 + 1)]);
      end loop;
      TS (EOS + 1) := To_Unbounded_String ("eos");
      S.Export_GGUF (M.all, Dir & "/student.gguf", TS, Bos => EOS, Eos => EOS, Ctx => Seq);
   end;
   Put_Line ("exported " & Dir & "/student.gguf");

   --  load engine, generate an SVG per held-out prefix
   declare
      LM  : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Dir & "/student.gguf");
      Vc  : constant Integer := LLM_Llama.Vocab_Size (LM);
      InF : File_Type; OutF : File_Type;

      function Gen (Prefix : String) return String is
         Ids : LLM_Tokenizer.Token_Array (1 .. Seq);
         Cur : Natural := 0;
         Res : Unbounded_String;
      begin
         for I in Prefix'Range loop
            exit when Cur >= Seq - 1;
            Cur := Cur + 1; Ids (Cur) := Character'Pos (Prefix (I));
         end loop;
         loop
            exit when Cur >= Seq;
            declare
               Fl  : constant LLM_Llama.Logits_Flat := LLM_Llama.Forward_Logits (LM, Ids (1 .. Cur));
               Bst : Integer := 0; BV : Float := Float'First;
            begin
               for T in 0 .. Vc - 1 loop
                  if Fl ((Cur - 1) * Vc + T) > BV then BV := Fl ((Cur - 1) * Vc + T); Bst := T; end if;
               end loop;
               exit when Bst = EOS;
               Cur := Cur + 1; Ids (Cur) := Bst;
               if Bst in 0 .. 255 then Res := Res & Character'Val (Bst); end if;
            end;
         end loop;
         return To_String (Res);
      end Gen;
   begin
      Open (InF, In_File, Dir & "/eval.txt");
      Create (OutF, Out_File, Dir & "/student_svgs.txt");
      while not End_Of_File (InF) loop
         declare Pfx : constant String := Get_Line (InF);
         begin
            if Pfx'Length > 0 then Put_Line (OutF, Gen (Pfx)); end if;
         end;
      end loop;
      Close (InF); Close (OutF);
   end;
   Put_Line ("wrote " & Dir & "/student_svgs.txt");
end Train_SVG;
