------------------------------------------------------------------------
-- test_qmatvec_guard — QMatVec rejects an input vector whose length does not
-- match the weight's in-dimension (would otherwise read out of bounds under
-- this unit's suppressed checks). Synthetic F32 weight; no model file needed.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_GGUF;
with LLM_Dequant;
with LLM_Tensor;            use LLM_Tensor;

procedure Test_QMatVec_Guard is
   Pass : Boolean := True;
   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   --  A tiny F32 weight: in=4, out=2 (row-major, 4*2*4 = 32 bytes, all zero).
   Info : constant LLM_GGUF.Tensor_Info :=
     (Name   => Null_Unbounded_String,
      N_Dims => 2,
      Dims   => [4, 2, 0, 0],
      Kind      => LLM_GGUF.GGML_TYPE_F32,
      Offset    => 0,
      Byte_Size => 0);
   Raw : constant String (1 .. 32) := [others => Character'Val (0)];

   Ok, R_Short, R_Long : Boolean := False;
begin
   Put_Line ("=== QMatVec input-length guard ===");

   --  Correct length (4) is accepted. (QMatVec is called in a nested block so
   --  an exception from its declaration reaches the outer handler.)
   declare
      X : constant Tensor := New_Tensor ([1, 4]);
   begin
      declare
         Y : constant Tensor := LLM_Dequant.QMatVec (Info, Raw, X);
         pragma Unreferenced (Y);
      begin null; end;
      Ok := True;
   exception when others => Ok := False;
   end;
   Check ("correct-length X (=in-dim) is accepted", Ok);

   --  Too-short X must raise, not read out of bounds.
   declare
      X : constant Tensor := New_Tensor ([1, 3]);
   begin
      declare
         Y : constant Tensor := LLM_Dequant.QMatVec (Info, Raw, X);
         pragma Unreferenced (Y);
      begin null; end;
   exception when Constraint_Error => R_Short := True;
   end;
   Check ("short X raises Constraint_Error", R_Short);

   --  Too-long X is also rejected (strict equality).
   declare
      X : constant Tensor := New_Tensor ([1, 8]);
   begin
      declare
         Y : constant Tensor := LLM_Dequant.QMatVec (Info, Raw, X);
         pragma Unreferenced (Y);
      begin null; end;
   exception when Constraint_Error => R_Long := True;
   end;
   Check ("over-long X raises Constraint_Error", R_Long);

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_QMatVec_Guard;
