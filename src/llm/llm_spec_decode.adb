---------------------------------------------------------------------
-- LLM_Spec_Decode body — see spec.
---------------------------------------------------------------------

pragma Warnings (Off, "formal parameter ""Params"" is not referenced");
--  Params carries Temperature/Seed for the sampling path (step 1 scope note in
--  the spec). The greedy path verified here ignores it by design; silencing the
--  warning keeps the signature stable for when acceptance sampling lands.

package body LLM_Spec_Decode is

   use LLM_Tokenizer;

   --  argmax over one vocab-wide logit row: row R of a [seq*vocab] flat buffer
   --  covers indices R*Vocab .. R*Vocab + Vocab - 1; returns the 0-based id.
   function Argmax_Row
     (L : LLM_Qwen.Logits_Flat; Row, Vocab : Natural) return Integer
   is
      Base : constant Natural := Row * Vocab;
      Best : Natural := 0;
      Bv   : Float := L (Base);
   begin
      for K in 1 .. Vocab - 1 loop
         if L (Base + K) > Bv then
            Bv := L (Base + K);
            Best := K;
         end if;
      end loop;
      return Best;
   end Argmax_Row;

   --  Greedy next-token from the LAST row of a forward over Ids.
   function Greedy_Next
     (M : LLM_Qwen.Qwen_Model; Ids : Token_Array; Vocab : Natural)
      return Integer
   is
      L : constant LLM_Qwen.Logits_Flat := LLM_Qwen.Forward_Logits (M, Ids);
      Last_Row : constant Natural := Ids'Length - 1;
   begin
      return Argmax_Row (L, Last_Row, Vocab);
   end Greedy_Next;

   function Generate
     (Draft, Target  : LLM_Qwen.Qwen_Model;
      Prompt_Ids     : LLM_Tokenizer.Token_Array;
      Max_New_Tokens : Positive;
      Stop_Id        : Integer;
      Gamma          : Positive := 4;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Result_Stats   : access Stats := null)
      return LLM_Tokenizer.Token_Array
   is
      Vocab : constant Natural := LLM_Qwen.Vocab_Size (Target);

      --  The running sequence: prompt then everything accepted/emitted. The
      --  draft and target both condition on this exact prefix.
      Seq : Token_Array (1 .. Prompt_Ids'Length + Max_New_Tokens);
      Len : Natural := Prompt_Ids'Length;

      Out_Ids : Token_Array (1 .. Max_New_Tokens);
      N_Out   : Natural := 0;

      St : Stats;

      procedure Bump_Out (Id : Integer) is
      begin
         N_Out := N_Out + 1;
         Out_Ids (N_Out) := Id;
         Len := Len + 1;
         Seq (Len) := Id;
         St.Emitted := St.Emitted + 1;
      end Bump_Out;

   begin
      Seq (1 .. Len) := Prompt_Ids;

      --  Greedy is the verified path (Temperature <= 0). Sampling would branch
      --  here into acceptance sampling; step 1 implements and tests greedy.
      Main : while N_Out < Max_New_Tokens loop

         --  1) DRAFT proposes up to Gamma tokens, autoregressively, each one
         --     conditioned on the prefix plus its own earlier proposals.
         declare
            G : constant Natural :=
              Natural'Min (Gamma, Max_New_Tokens - N_Out);
            Draft_Ids : Token_Array (1 .. G);
            Work      : Token_Array (1 .. Len + G);
            WLen      : Natural := Len;
         begin
            Work (1 .. Len) := Seq (1 .. Len);
            for I in 1 .. G loop
               declare
                  D : constant Integer :=
                    Greedy_Next (Draft, Work (1 .. WLen), Vocab);
               begin
                  St.Draft_Forwards := St.Draft_Forwards + 1;
                  Draft_Ids (I) := D;
                  WLen := WLen + 1;
                  Work (WLen) := D;
               end;
            end loop;
            St.Proposed := St.Proposed + G;

            --  2) TARGET verifies all G proposals in ONE pass over
            --     [prefix, d_1..d_G]. Row (Len-1) predicts d_1, row (Len)d_2,
            --     ... row (Len+G-1) is the bonus token after the last draft.
            declare
               --  Capture the prefix length BEFORE any accept mutates Len:
               --  Bump_Out advances Len, and every row index below is measured
               --  from this fixed base. Using the live Len would drift by one
               --  per accepted token — a silent off-by-one that corrupts the
               --  acceptance check.
               Base_Len  : constant Natural := Len;
               Verify_In : constant Token_Array := Seq (1 .. Base_Len) & Draft_Ids;
               TL : constant LLM_Qwen.Logits_Flat :=
                 LLM_Qwen.Forward_Logits (Target, Verify_In);
               Accepted_Here : Natural := 0;
               Mismatch_Tok  : Integer := -1;
            begin
               St.Target_Forwards := St.Target_Forwards + 1;

               --  3) Accept the longest prefix where the target's greedy pick
               --     equals the draft's. Rows are 0-based; the prompt occupies
               --     rows 0 .. Base_Len-1, so the target's prediction FOR draft
               --     token I (given prompt + d_1..d_{I-1}) is row Base_Len-1+(I-1).
               for I in 1 .. G loop
                  declare
                     T : constant Integer :=
                       Argmax_Row (TL, Base_Len - 1 + (I - 1), Vocab);
                  begin
                     if T = Draft_Ids (I) then
                        Bump_Out (T);
                        Accepted_Here := Accepted_Here + 1;
                        exit Main when T = Stop_Id or else N_Out >= Max_New_Tokens;
                     else
                        --  First disagreement: the target's own token replaces
                        --  the draft's, and the rest of the draft is discarded.
                        Mismatch_Tok := T;
                        exit;
                     end if;
                  end;
               end loop;

               St.Accepted := St.Accepted + Accepted_Here;

               --  4) One extra token always comes free from the target: either
               --     its correction at the first mismatch, or (whole draft
               --     accepted) the bonus at the last verify row, Base_Len-1+G,
               --     which predicts the token after d_G.
               if N_Out < Max_New_Tokens then
                  if Mismatch_Tok >= 0 then
                     Bump_Out (Mismatch_Tok);
                     exit Main when Mismatch_Tok = Stop_Id;
                  elsif Accepted_Here = G then
                     declare
                        Bonus : constant Integer :=
                          Argmax_Row (TL, Base_Len - 1 + G, Vocab);
                     begin
                        Bump_Out (Bonus);
                        exit Main when Bonus = Stop_Id;
                     end;
                  end if;
               end if;
            end;
         end;
      end loop Main;

      if Result_Stats /= null then
         Result_Stats.all := St;
      end if;
      return Out_Ids (1 .. N_Out);
   end Generate;

end LLM_Spec_Decode;
