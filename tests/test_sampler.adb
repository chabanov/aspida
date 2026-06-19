------------------------------------------------------------------------
-- test_sampler — deterministic checks for LLM_Sampler, focused on min-p.
-- With strictly-decreasing logits [5,4,3,2,1,0,-1,-2] the softmax ratios to
-- the max are 1, e^-1(0.368), e^-2(0.135), e^-3(0.050)... so min-p's keep-set
-- is exactly predictable: every draw must come from the surviving tokens.
------------------------------------------------------------------------

with Ada.Text_IO;   use Ada.Text_IO;
with LLM_Sampler;
with LLM_Tensor;    use LLM_Tensor;

procedure Test_Sampler is
   Pass : Boolean := True;
   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   N    : constant := 8;
   Lg   : Tensor := New_Tensor ([1, N]);
   Vals : constant array (1 .. N) of Float := [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0];

   --  Draw Draws times; return the highest token id (0-based) ever produced
   --  and whether every draw stayed within the allowed set 0 .. Max_Allowed.
   procedure Sweep (P : LLM_Sampler.Params; Draws : Integer;
                    Max_Allowed : Integer; Within : out Boolean;
                    Hi : out Integer) is
      S : LLM_Sampler.Sampler := LLM_Sampler.Create (P);
   begin
      Within := True; Hi := 0;
      for I in 1 .. Draws loop
         declare T : constant Integer := LLM_Sampler.Next (S, Lg); begin
            if T > Hi then Hi := T; end if;
            if T < 0 or else T > Max_Allowed then Within := False; end if;
         end;
      end loop;
   end Sweep;

begin
   Put_Line ("=== LLM_Sampler ===");
   for I in 1 .. N loop Set_Flat (Lg, I, Vals (I)); end loop;

   --  Greedy: argmax is token 0.
   declare
      S : LLM_Sampler.Sampler := LLM_Sampler.Create (LLM_Sampler.Greedy);
   begin
      Check ("greedy returns argmax (token 0)", LLM_Sampler.Next (S, Lg) = 0);
   end;

   declare W : Boolean; H : Integer;
   begin
      --  min_p = 0.99: only token 0 (ratio 1.0) survives -> always argmax.
      Sweep ((Temperature => 1.0, Min_P => 0.99, Seed => 7, others => <>),
             3000, 0, W, H);
      Check ("min_p=0.99 collapses to argmax (only token 0)", W and then H = 0);

      --  min_p = 0.3: keep token0(1.0) + token1(0.368); token2(0.135) excluded.
      Sweep ((Temperature => 1.0, Min_P => 0.3, Seed => 7, others => <>),
             3000, 1, W, H);
      Check ("min_p=0.3 keeps {0,1}, never a lower-prob token", W);
      Check ("min_p=0.3 actually samples token 1 too", H = 1);

      --  min_p = 0.1: keep token0,1,2 (0.135 >= 0.1); token3(0.050) excluded.
      Sweep ((Temperature => 1.0, Min_P => 0.1, Seed => 7, others => <>),
             3000, 2, W, H);
      Check ("min_p=0.1 keeps {0,1,2}, never token >=3", W);
      Check ("min_p=0.1 reaches token 2", H = 2);

      --  min_p off (=0) with pure temperature: full vocab reachable, all valid.
      Sweep ((Temperature => 1.0, Seed => 7, others => <>), 5000, N - 1, W, H);
      Check ("min_p off: draws stay in range and reach the tail", W and then H >= 5);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Sampler;
