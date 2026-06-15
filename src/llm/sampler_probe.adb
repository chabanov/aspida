--  Sampler_Probe — fast unit checks for LLM_Sampler (no model needed).
with Ada.Text_IO; use Ada.Text_IO;
with LLM_Tensor; use LLM_Tensor;
with LLM_Sampler;

procedure Sampler_Probe is
   Fails : Natural := 0;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      Put_Line ((if Cond then "  ok   " else "  FAIL ") & Name);
      if not Cond then Fails := Fails + 1; end if;
   end Check;

   --  logits: index 3 (id 2) is the clear max.
   L : Tensor := New_Tensor ([1, 5]);
begin
   Set_Flat (L, 1, 0.5); Set_Flat (L, 2, 1.0); Set_Flat (L, 3, 4.0);
   Set_Flat (L, 4, 0.2); Set_Flat (L, 5, -1.0);

   declare
      Sg : LLM_Sampler.Sampler := LLM_Sampler.Create (LLM_Sampler.Greedy);
   begin
      Check ("greedy picks argmax (id 2)", LLM_Sampler.Next (Sg, L) = 2);
   end;

   declare
      Sk : LLM_Sampler.Sampler := LLM_Sampler.Create
        ((Temperature => 1.0, Top_K => 1, others => <>));
   begin
      Check ("top_k=1 == argmax (id 2)", LLM_Sampler.Next (Sk, L) = 2);
   end;

   --  Determinism: same seed + same logits => identical draws.
   declare
      P  : constant LLM_Sampler.Params :=
        (Temperature => 1.5, Top_K => 5, Top_P => 0.95, Seed => 42, others => <>);
      A  : LLM_Sampler.Sampler := LLM_Sampler.Create (P);
      B  : LLM_Sampler.Sampler := LLM_Sampler.Create (P);
      Eq : Boolean := True;
   begin
      for I in 1 .. 50 loop
         if LLM_Sampler.Next (A, L) /= LLM_Sampler.Next (B, L) then Eq := False; end if;
      end loop;
      Check ("same seed is reproducible", Eq);
   end;

   --  A high-temperature sampler should sometimes pick a non-argmax token.
   declare
      Sh : LLM_Sampler.Sampler := LLM_Sampler.Create
        ((Temperature => 3.0, Top_K => 5, Top_P => 1.0, Seed => 7, others => <>));
      Saw_Other : Boolean := False;
   begin
      for I in 1 .. 200 loop
         if LLM_Sampler.Next (Sh, L) /= 2 then Saw_Other := True; end if;
      end loop;
      Check ("temperature explores beyond argmax", Saw_Other);
   end;

   --  Repetition penalty should suppress a repeated token (id 2) under greedy.
   declare
      Sr  : LLM_Sampler.Sampler := LLM_Sampler.Create
        ((Temperature => 0.0, Repeat_Penalty => 100.0, others => <>));
      Hist : constant LLM_Sampler.History := (1 => 2);  -- id 2 recently used
   begin
      Check ("repeat penalty avoids id 2", LLM_Sampler.Next (Sr, L, Hist) /= 2);
   end;

   New_Line;
   if Fails = 0 then Put_Line ("ALL SAMPLER CHECKS PASSED");
   else Put_Line (Fails'Image & " CHECK(S) FAILED"); end if;
end Sampler_Probe;
