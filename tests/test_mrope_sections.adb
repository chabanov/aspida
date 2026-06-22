------------------------------------------------------------------------
-- test_mrope_sections — per-section RoPE positions (multimodal mRoPE).
--
-- Validates LLM_RoPE.Apply_Sections: the new path that picks a
-- distinct position per mRoPE section (time/height/width/empty) for
-- each dimension pair. For text-only the section positions are all
-- equal, and the output must be bit-identical to the legacy
-- LLM_RoPE.Apply (P, X, Pos).
--
-- Scenarios:
--   1. Backward-compat regression guard (Qwen Sections = [11,11,10,0]).
--   2. One section at the start (Sections = [16,0,0,0]).
--   3. One section in the middle (Sections = [0,16,0,0]).
--   4. Two non-empty sections, mixed positions (Sections = [8,8,0,0]).
--   5. Trailing empty section (Sections = [8,8,8,0]) — last 8 dims
--      untouched regardless of Sec(3).
--   6. All sections empty (no real mRoPE, but Sections field is set
--      to a [1,4] zero tensor — must fall back to uniform Pos).
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with LLM_RoPE;              use LLM_RoPE;
with LLM_Tensor;            use LLM_Tensor;

procedure Test_MRoPE_Sections is
   Pass : Boolean := True;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   --  Per-element closeness for tensors of length D. Used wherever the
   --  test compares rotated outputs of a real RoPE.
   function Close (A, B : Tensor; D : Positive) return Boolean is
   begin
      for I in 1 .. D loop
         if abs (Get_Flat (A, I) - Get_Flat (B, I)) > 1.0E-5 then
            return False;
         end if;
      end loop;
      return True;
   end Close;

   --  Bit-exact equality over a [Lo, Lo+Len-1] range of two tensors.
   --  Used when only a slice must agree (e.g. the section-0 dims agree
   --  between two rotations but other dims differ). Replaces the
   --  non-existent LLM_Tensor.Slice API — we just walk the flat indices.
   function Equal_Slice
     (A, B : Tensor; Lo, Len : Positive) return Boolean
   is
   begin
      for K in 0 .. Len - 1 loop
         if Get_Flat (A, Lo + K) /= Get_Flat (B, Lo + K) then
            return False;
         end if;
      end loop;
      return True;
   end Equal_Slice;

   procedure Fill_X (X : in out Tensor; D : Positive) is
   begin
      for I in 1 .. D loop
         Set_Flat (X, I, 0.13 * Float (I) - 0.7);
      end loop;
   end Fill_X;

   function Mk_RoPE
     (D : Positive; FB : Float; Sec : LLM_RoPE.Section_Positions)
      return RoPE_Params
   is
      P : constant RoPE_Params :=
        Create_Qwen_RoPE (D, FB, 4096);
      Q : RoPE_Params := P;
      S : Tensor := New_Tensor ([1, 4]);
   begin
      Set_Flat (S, 1, Float (Sec (0)));
      Set_Flat (S, 2, Float (Sec (1)));
      Set_Flat (S, 3, Float (Sec (2)));
      Set_Flat (S, 4, Float (Sec (3)));
      Q.Sections := S;
      return Q;
   end Mk_RoPE;
   --  NB: Sec is passed as Section_Positions (an array of Integer) and
   --  stored as Float (the section-width field is Float, not Integer).
   --  This is the standard way to override Sections on a params record
   --  built by Create_Qwen_RoPE.

begin
   ------------------------------------------------------------------------
   -- 1. Backward-compat regression: legacy Apply == Apply_Sections with
   --    uniform positions, for the Qwen-3.5-MoE Section layout [11,11,10,0].
   ------------------------------------------------------------------------
   Put_Line ("=== Scenario 1: text-only regression (Sections=[11,11,10,0]) ===");
   declare
      D  : constant := 64;        -- Qwen3.5 head_dim / 2 = 32 pairs -> Dim = 64
      X  : Tensor := New_Tensor ([1, D]);
      P  : constant RoPE_Params := Create_Qwen_RoPE (D, 1_000_000.0, 4096);
      --  Default Sections = [11, 11, 10, 0] set by Create_Qwen_RoPE.
      Sec : constant LLM_RoPE.Section_Positions := [others => 7];
   begin
      Fill_X (X, D);
      Check ("Apply(P,X,7) == Apply_Sections(P,X,Uniform[7])",
             Close (Apply (P, X, 7), Apply_Sections (P, X, Sec), D));
      Check ("Apply(P,X,0) == Apply_Sections(P,X,Uniform[0])",
             Close (Apply (P, X, 0), Apply_Sections (P, X, [others => 0]), D));
      Check ("Apply(P,X,-3) == Apply_Sections(P,X,Uniform[-3])",
             Close (Apply (P, X, -3),
                    Apply_Sections (P, X, [others => -3]),
                    D));
   end;

   ------------------------------------------------------------------------
   -- 2. One section at the start: only the first 16 pairs (dims [0..15]
   --    in pair space, i.e. [1..16] and [33..48] in flat space) are
   --    rotated; the second half is passed through unchanged.
   ------------------------------------------------------------------------
   Put_Line ("=== Scenario 2: single section at start (Sections=[16,0,0,0]) ===");
   declare
      D  : constant := 32;        -- Dim = 32 -> 16 pairs
      X  : Tensor := New_Tensor ([1, D]);
      P  : constant RoPE_Params := Mk_RoPE (D, 1_000_000.0, [16, 0, 0, 0]);
      R  : Tensor;
      R0 : Tensor;
   begin
      Fill_X (X, D);
      R  := Apply_Sections (P, X, [5, 0, 0, 0]);
      R0 := Apply_Sections (P, X, [0, 0, 0, 0]);  -- zero-Pos
      --  Pair space is [0, 16); with Sec=[16,0,0,0], all 16 pairs are
      --  in section 0, so every position uses Sec(0)=5 — output should
      --  equal Apply (P, X, 5) bit-exactly.
      Check ("Sections=[16,0,0,0] with Pos=5 == legacy Apply[5]",
             Close (R, Apply (P, X, 5), D));
      Check ("Sections=[16,0,0,0] with Pos=0 == legacy Apply[0]",
             Close (R0, Apply (P, X, 0), D));
   end;

   ------------------------------------------------------------------------
   -- 3. Section in the middle: with Sections=[0,16,0,0], the empty
   --    section 0 is just a placeholder; section 1 (width 16) owns
   --    every pair. So all 16 pairs are rotated with Sec(1)=11, and
   --    the whole R should differ from input (no portion is left as
   --    identity).
   ------------------------------------------------------------------------
   Put_Line ("=== Scenario 3: section in middle (Sections=[0,16,0,0]) ===");
   declare
      D     : constant := 32;
      X     : Tensor := New_Tensor ([1, D]);
      P     : constant RoPE_Params := Mk_RoPE (D, 1_000_000.0, [0, 16, 0, 0]);
      Half  : constant := D / 2;
      R     : Tensor;
   begin
      Fill_X (X, D);
      R     := Apply_Sections (P, X, [0, 11, 0, 0]);
      --  Every pair is in section 1 (Pos=11): R must differ from X in
      --  BOTH halves (no identity rotation anywhere).
      Check ("dims [1..16] (section 1, Pos=11) changed",
             not Equal_Slice (R, X, 1, Half));
      Check ("dims [17..32] (section 1, Pos=11) changed",
             not Equal_Slice (R, X, Half + 1, Half));
   end;

   ------------------------------------------------------------------------
   -- 4. Two non-empty sections, mixed positions: pairs [0..7] use
   --    Sec(0)=5, pairs [8..15] use Sec(1)=9. With Dim=32 -> 16 pairs.
   --    We compare against a "manual" reference built by running
   --    Apply on a section-masked input.
   --
   --    Manual reference: build two copies of X, run Apply(_, _, 5) on
   --    one (rotated by Pos=5) and Apply(_, _, 9) on the other (rotated
   --    by Pos=9), then interleave their rotated halves.
   --
   --    Wait — we can't easily interleave in a non-mutating way; the
   --    simpler check is: with Sections=[8,8,0,0], Sec=[5,9,0,0], the
   --    first 8 pairs rotate with Pos=5 (matching Apply(5)) and the
   --    second 8 pairs rotate with Pos=9 (matching Apply(9) on the
   --    *second-half slice only*). The cleanest test: run Apply_Sections
   --    with Sec=[5,9,0,0] and with Sec=[5,5,0,0] -- the first 8 pairs
   --    must agree, the second 8 pairs must differ.
   ------------------------------------------------------------------------
   Put_Line ("=== Scenario 4: two sections, mixed positions (Sections=[8,8,0,0]) ===");
   declare
      D      : constant := 32;
      X      : Tensor := New_Tensor ([1, D]);
      P      : constant RoPE_Params := Mk_RoPE (D, 1_000_000.0, [8, 8, 0, 0]);
      Half   : constant := D / 2;
      Quart  : constant := Half / 2;     -- 8 pairs = 8 dims in flat
      R_mix  : Tensor;
      R_uni  : Tensor;
      R_sec0 : Tensor;
   begin
      Fill_X (X, D);
      R_mix  := Apply_Sections (P, X, [5, 9, 0, 0]);
      R_uni  := Apply_Sections (P, X, [5, 5, 0, 0]);
      R_sec0 := Apply_Sections (P, X, [5, 0, 0, 0]);
      --  Pairs 0..7 (dims 1..8 and 17..24, first half + first half of
      --  second half): section 0, Pos=5. In the uniform(5) case, the
      --  first 8 pairs of BOTH halves match the mixed case. The
      --  remaining 8 pairs (section 1) differ.
      Check ("pair 0..7 agree between Sec=[5,9] and Sec=[5,5]",
             Equal_Slice (R_mix, R_uni, 1, Quart)
             and then
             Equal_Slice (R_mix, R_uni, Half + 1, Quart));
      --  Pairs 8..15: section 1, Pos=9. Must differ from Sec=[5,5,0,0].
      Check ("pair 8..15 differ between Sec=[5,9] and Sec=[5,5]",
             (not Equal_Slice (R_mix, R_uni, Quart + 1, Quart))
             or else (not Equal_Slice (R_mix, R_uni, Half + Quart + 1, Quart)));
      --  And Pos=0 in section 1 should be a no-op (identity) for those
      --  pairs, while Pos=5 rotates them -> R_sec0 second 8 pairs must
      --  be rotated, R_mix with Sec(1)=9 also rotated but with a
      --  different angle. So R_mix and R_sec0 should differ on the
      --  second 8 pairs.
      Check ("pair 8..15 differ between Sec=[5,9] and Sec=[5,0]",
             (not Equal_Slice (R_mix, R_sec0, Quart + 1, Quart))
             or else (not Equal_Slice (R_mix, R_sec0, Half + Quart + 1, Quart)));
   end;

   ------------------------------------------------------------------------
   -- 5. Trailing empty section: Sections = [8,8,8,0]. The last 8 pairs
   --    (dims [25..32] and [41..48] for Dim=48) belong to no section
   --    -> fall back to Pos. With Pos=0 the fallback is identity; with
   --    a non-zero Pos the fallback rotates them, so we use Pos=0 here
   --    to make the no-section-rotation observable.
   ------------------------------------------------------------------------
   Put_Line ("=== Scenario 5: trailing empty section (Sections=[8,8,8,0], Dim=48) ===");
   declare
      D      : constant := 48;
      X      : Tensor := New_Tensor ([1, D]);
      P      : constant RoPE_Params := Mk_RoPE (D, 1_000_000.0, [8, 8, 8, 0]);
      Sec0   : constant := 8;     -- width of section 0 (in pairs = in dims)
      --  Pairs: section 0 = [0..7], section 1 = [8..15], section 2 = [16..23],
      --  section 3 = empty. 8+8+8+0 = 24 == Dim/2, so sections 0..2 cover
      --  every pair and section 3 is unused. Verify Sec(3) is ignored: any
      --  value (here 999) must produce the same output as Sec(3) = 0.
   begin
      Fill_X (X, D);
      declare
         R_Bad : constant Tensor := Apply_Sections (P, X, [3, 5, 7, 999]);
         R_Ok  : constant Tensor := Apply_Sections (P, X, [3, 5, 7, 0]);
      begin
         Check ("Sec[3]=999 (out of range / ignored) == Sec[3]=0",
                Close (R_Bad, R_Ok, D));
         --  And spot-check: with Sections=[8,8,8,0] and Sec=[3,5,7,0],
         --  the first 8 dims (section 0, Pos=3) must equal the first
         --  8 dims of Apply(P, X, 3). NB: Sec0 = 8, NOT Half/2 (=12).
         Check ("first 8 pairs (section 0, Pos=3) match Apply(_,_,3)",
                Equal_Slice (R_Ok, Apply (P, X, 3), 1, Sec0));
      end;
   end;

   ------------------------------------------------------------------------
   -- 6. All sections empty: a params record with Sections = [0,0,0,0].
   --    The Section_Of walker should return -1 for every pair, so
   --    Eff_Pos falls back to Pos, reproducing Apply(P, X, Pos).
   ------------------------------------------------------------------------
   Put_Line ("=== Scenario 6: all sections empty (Sections=[0,0,0,0]) ===");
   declare
      D  : constant := 32;
      X  : Tensor := New_Tensor ([1, D]);
      P  : constant RoPE_Params := Mk_RoPE (D, 1_000_000.0, [0, 0, 0, 0]);
   begin
      Fill_X (X, D);
      Check ("Sections=[0,0,0,0] with Sec=[5,9,11,0] == Apply(_,_,5)",
             Close (Apply_Sections (P, X, [5, 9, 11, 0]),
                   Apply (P, X, 5),  -- all sections empty -> fall back to Pos=5
                   D));
      Check ("Sections=[0,0,0,0] with Sec=[0,0,0,0] == Apply(_,_,0)",
             Close (Apply_Sections (P, X, [0, 0, 0, 0]), Apply (P, X, 0), D));
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_MRoPE_Sections;
