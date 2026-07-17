---------------------------------------------------------------------
-- LLM_Spec_Decode body — see spec.
--
-- Steps 2-4 (target verifies, accept longest matching prefix, emit one free
-- correction/bonus) are the PROVEN core and live in Verify_Round. The two
-- drivers differ only in step 1 — how the Gamma proposals are produced:
--   Generate        : a draft MODEL, autoregressively.
--   Generate_Lookup : the CONTEXT, by n-gram match (no second model).
-- Both are byte-identical to target-alone greedy because the target verifies
-- every proposed token; the draft SOURCE only changes the acceptance rate,
-- never the output. test_spec_decode asserts that for both.
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
   begin
      return Argmax_Row (L, Ids'Length - 1, Vocab);
   end Greedy_Next;

   --  Prompt-lookup draft: find the most recent EARLIER occurrence of the last
   --  Ngram tokens of Seq (1 .. Len) and return up to Gamma tokens that
   --  followed it. Empty when there is no repeat — the caller then falls back
   --  to a plain one-token target step. This is the whole "draft model": a
   --  search over tokens already in context, so it costs no forward and no VRAM.
   function Lookup_Draft
     (Seq : Token_Array; Len, Ngram, Gamma : Natural) return Token_Array
   is
   begin
      if Len < Ngram + 1 then
         return Seq (1 .. 0);   -- not enough history to match an n-gram
      end if;
      --  The suffix we are matching sits at Seq (Len-Ngram+1 .. Len). Scan
      --  backwards for the freshest earlier position P whose n-gram equals it
      --  AND that has at least one following token (P + Ngram <= Len).
      for P in reverse Seq'First .. Len - Ngram loop
         declare
            Match : Boolean := True;
         begin
            for K in 0 .. Ngram - 1 loop
               if Seq (P + K) /= Seq (Len - Ngram + 1 + K) then
                  Match := False;
                  exit;
               end if;
            end loop;
            if Match then
               declare
                  --  Tokens that followed the match: Seq (P+Ngram ..), clipped
                  --  to what exists and to Gamma.
                  Avail : constant Natural :=
                    Natural'Min (Gamma, Len - (P + Ngram) + 1);
               begin
                  if Avail = 0 then
                     return Seq (1 .. 0);
                  end if;
                  return Seq (P + Ngram .. P + Ngram + Avail - 1);
               end;
            end if;
         end;
      end loop;
      return Seq (1 .. 0);   -- no earlier occurrence
   end Lookup_Draft;

   --  Shared verify core (steps 2-4). Verifies Draft_Ids in one target forward,
   --  appends the accepted prefix plus one correction/bonus to Out_Ids, and
   --  advances Seq/Len. Draft_Ids'Length = 0 is valid and degenerates to a
   --  single plain greedy target step (the fallback when lookup finds nothing).
   --  Stopped is set when a stop id or the token cap is hit.
   procedure Verify_Round
     (Target    : LLM_Qwen.Qwen_Model;
      Seq       : in out Token_Array;
      Len       : in out Natural;
      Draft_Ids : Token_Array;
      Vocab     : Natural;
      Max_New   : Positive;
      Stop_Id   : Integer;
      Out_Ids   : in out Token_Array;
      N_Out     : in out Natural;
      St        : in out Stats;
      Stopped   : out Boolean)
   is
      G        : constant Natural := Draft_Ids'Length;
      Base_Len : constant Natural := Len;   -- fixed before any accept mutates Len
      --  For G = 0 the verify input is just the prefix; its last row predicts
      --  the next token, which the step-4 "whole draft accepted" branch emits.
      Verify_In : constant Token_Array :=
        (if G = 0 then Seq (1 .. Base_Len) else Seq (1 .. Base_Len) & Draft_Ids);
      TL : constant LLM_Qwen.Logits_Flat :=
        LLM_Qwen.Forward_Logits (Target, Verify_In);
      Accepted_Here : Natural := 0;
      Mismatch_Tok  : Integer := -1;

      procedure Bump_Out (Id : Integer) is
      begin
         N_Out := N_Out + 1;
         Out_Ids (N_Out) := Id;
         Len := Len + 1;
         Seq (Len) := Id;
         St.Emitted := St.Emitted + 1;
      end Bump_Out;
   begin
      Stopped := False;
      St.Target_Forwards := St.Target_Forwards + 1;
      St.Proposed := St.Proposed + G;

      --  Accept the longest prefix where the target's greedy pick equals the
      --  draft's. J is a 0-based offset so this is INDEPENDENT of Draft_Ids'
      --  bounds: Generate passes a 1-based array, but Generate_Lookup passes a
      --  raw slice of Seq whose 'First is wherever the match was found. Indexing
      --  Draft_Ids (Draft_Ids'First + J) works for both; the earlier
      --  1-based-only Draft_Ids (I) crashed the lookup path (index check).
      --  Rows are 0-based; the prompt occupies rows 0 .. Base_Len-1, so the
      --  target's prediction for draft offset J is row Base_Len-1+J.
      for J in 0 .. G - 1 loop
         declare
            T : constant Integer :=
              Argmax_Row (TL, Base_Len - 1 + J, Vocab);
         begin
            if T = Draft_Ids (Draft_Ids'First + J) then
               Bump_Out (T);
               Accepted_Here := Accepted_Here + 1;
               if T = Stop_Id or else N_Out >= Max_New then
                  Stopped := True;
                  St.Accepted := St.Accepted + Accepted_Here;
                  return;
               end if;
            else
               Mismatch_Tok := T;   -- target's own token replaces the draft
               exit;
            end if;
         end;
      end loop;

      St.Accepted := St.Accepted + Accepted_Here;

      --  One extra token always comes free from the target: its correction at
      --  the first mismatch, or (whole draft accepted, incl. the G=0 case) the
      --  bonus at row Base_Len-1+G, which predicts the token after the draft.
      if N_Out < Max_New then
         if Mismatch_Tok >= 0 then
            Bump_Out (Mismatch_Tok);
            Stopped := Mismatch_Tok = Stop_Id;
         elsif Accepted_Here = G then
            declare
               Bonus : constant Integer :=
                 Argmax_Row (TL, Base_Len - 1 + G, Vocab);
            begin
               Bump_Out (Bonus);
               Stopped := Bonus = Stop_Id;
            end;
         end if;
      end if;
   end Verify_Round;

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
      Seq   : Token_Array (1 .. Prompt_Ids'Length + Max_New_Tokens);
      Len   : Natural := Prompt_Ids'Length;
      Out_Ids : Token_Array (1 .. Max_New_Tokens);
      N_Out   : Natural := 0;
      St      : Stats;
   begin
      Seq (1 .. Len) := Prompt_Ids;
      while N_Out < Max_New_Tokens loop
         declare
            G : constant Natural := Natural'Min (Gamma, Max_New_Tokens - N_Out);
            Draft_Ids : Token_Array (1 .. G);
            Work      : Token_Array (1 .. Len + G);
            WLen      : Natural := Len;
            Stopped   : Boolean;
         begin
            --  Draft: the model proposes G tokens autoregressively.
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
            Verify_Round (Target, Seq, Len, Draft_Ids, Vocab, Max_New_Tokens,
                          Stop_Id, Out_Ids, N_Out, St, Stopped);
            exit when Stopped;
         end;
      end loop;
      if Result_Stats /= null then
         Result_Stats.all := St;
      end if;
      return Out_Ids (1 .. N_Out);
   end Generate;

   function Generate_Lookup
     (Target         : LLM_Qwen.Qwen_Model;
      Prompt_Ids     : LLM_Tokenizer.Token_Array;
      Max_New_Tokens : Positive;
      Stop_Id        : Integer;
      Ngram          : Positive := 3;
      Gamma          : Positive := 8;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Result_Stats   : access Stats := null)
      return LLM_Tokenizer.Token_Array
   is
      Vocab : constant Natural := LLM_Qwen.Vocab_Size (Target);
      Seq   : Token_Array (1 .. Prompt_Ids'Length + Max_New_Tokens);
      Len   : Natural := Prompt_Ids'Length;
      Out_Ids : Token_Array (1 .. Max_New_Tokens);
      N_Out   : Natural := 0;
      St      : Stats;
   begin
      Seq (1 .. Len) := Prompt_Ids;
      while N_Out < Max_New_Tokens loop
         declare
            Cap : constant Natural := Natural'Min (Gamma, Max_New_Tokens - N_Out);
            --  Draft: copy from context. Empty when nothing repeats -> Verify_Round
            --  degenerates to one plain greedy target step, so the loop always
            --  makes progress and the output is identical either way.
            Draft_Ids : constant Token_Array :=
              Lookup_Draft (Seq, Len, Ngram, Cap);
            Stopped   : Boolean;
         begin
            Verify_Round (Target, Seq, Len, Draft_Ids, Vocab, Max_New_Tokens,
                          Stop_Id, Out_Ids, N_Out, St, Stopped);
            exit when Stopped;
         end;
      end loop;
      if Result_Stats /= null then
         Result_Stats.all := St;
      end if;
      return Out_Ids (1 .. N_Out);
   end Generate_Lookup;

end LLM_Spec_Decode;
