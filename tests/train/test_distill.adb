------------------------------------------------------------------------
-- test_distill — proves the teacher->student dataset pipeline:
--   * Capture keeps exactly the top-K largest teacher logits per position,
--     in descending order;
--   * the dataset round-trips through disk byte-for-byte;
--   * Teacher_Prob reconstructs a valid probability distribution.
-- Uses a deterministic synthetic teacher (no model load needed).
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Distill;     use Distill;

procedure Test_Distill is
   Pass : Boolean := True;
   VV   : constant := 64;          -- synthetic vocabulary
   KK   : constant := 8;           -- top-K kept

   --  Synthetic teacher: each row peaks at id = Tokens(row) mod V, with a
   --  strictly-decreasing-by-distance logit (distinct values, clear top-K).
   type Synth is new Teacher with record V : Positive := VV; end record;
   overriding function Vocab (T : Synth) return Positive is (T.V);
   overriding procedure Forward
     (T : in out Synth; Tokens : Token_Array; Out_Logits : out Logit_Matrix)
   is
      Peak : Integer;
   begin
      for R in 1 .. Tokens'Length loop
         Peak := Integer (Tokens (Tokens'First + R - 1)) mod T.V + 1;
         for C in 1 .. T.V loop
            Out_Logits (R, C) :=
              Logit (-0.1 * Float (abs (C - Peak)) - 0.0001 * Float (C));
         end loop;
      end loop;
   end Forward;

   S    : Synth;
   Toks : constant Token_Array := [5, 17, 40, 2, 63, 31];
   Smp  : constant Sample := Capture (S, Toks, KK);
   D    : Dataset;
   Path : constant String := "/tmp/aspida_distill_test.bin";
begin
   Put_Line ("=== Aspida distillation-dataset self-test ===");

   --  [1] top-K correctness vs a direct full-row scan
   declare
      L : Logit_Matrix (1 .. Toks'Length, 1 .. VV);
      Ok_TopK, Ok_Order : Boolean := True;
   begin
      Forward (S, Toks, L);
      for R in 1 .. Toks'Length loop
         --  selected logits must be descending
         for J in 1 .. KK - 1 loop
            if Smp.Top_Logit (R, J) < Smp.Top_Logit (R, J + 1) then
               Ok_Order := False;
            end if;
         end loop;
         --  no unselected id may exceed the smallest selected logit
         declare
            Min_Sel : constant Logit := Smp.Top_Logit (R, KK);
            Selected : array (1 .. VV) of Boolean := [others => False];
         begin
            for J in 1 .. KK loop
               Selected (Integer (Smp.Top_Ids (R, J)) + 1) := True;
            end loop;
            for C in 1 .. VV loop
               if not Selected (C) and then L (R, C) > Min_Sel then
                  Ok_TopK := False;
               end if;
            end loop;
         end;
      end loop;
      Put_Line ("  top-K = K largest per row: " & Ok_TopK'Image);
      Put_Line ("  top-K descending order:    " & Ok_Order'Image);
      if not (Ok_TopK and Ok_Order) then Pass := False; end if;
   end;

   --  [2] disk round-trip (byte-for-byte)
   D.Append (Smp);
   D.Append (Capture (S, Token_Array'[1, 2, 3], 4));
   Write (Path, D);
   declare
      D2   : constant Dataset := Read (Path);
      Same : Boolean := True;
   begin
      if Natural (D2.Length) /= Natural (D.Length) then
         Same := False;
      else
         for I in 1 .. Natural (D.Length) loop
            declare
               A : Sample renames D (I);
               B : Sample renames D2 (I);
            begin
               if A.N /= B.N or else A.K /= B.K then
                  Same := False;
               else
                  for R in 1 .. A.N loop
                     if A.Tokens (R) /= B.Tokens (R) then Same := False; end if;
                     for J in 1 .. A.K loop
                        if A.Top_Ids (R, J) /= B.Top_Ids (R, J)
                          or else A.Top_Logit (R, J) /= B.Top_Logit (R, J)
                        then
                           Same := False;
                        end if;
                     end loop;
                  end loop;
               end if;
            end;
         end loop;
      end if;
      Put_Line ("  dataset round-trips through disk: " & Same'Image);
      if not Same then Pass := False; end if;
   end;

   --  [3] teacher distribution is a valid, peaked probability vector
   declare
      P   : constant Prob_Vector := Teacher_Prob (Smp, 1);
      Sum : Long_Float := 0.0;
      Ok  : Boolean := True;
   begin
      for J in P'Range loop Sum := Sum + P (J); end loop;
      if abs (Sum - 1.0) > 1.0E-9 then Ok := False; end if;        -- normalized
      for J in 1 .. KK - 1 loop
         if P (J) < P (J + 1) then Ok := False; end if;             -- descending
      end loop;
      Put_Line ("  Teacher_Prob sum =" & Sum'Image
                & "  P(top) =" & P (1)'Image);
      if not Ok then Pass := False; end if;
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Distill;
