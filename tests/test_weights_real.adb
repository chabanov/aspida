---------------------------------------------------------------------
-- Validate the GGUF load + dequant path on REAL model tensors.
--
-- Loads a few tensors from the model (F32 norms/router and Q8_K experts)
-- and checks the dequantised values are sane: finite, weight-magnitude,
-- and (for Q8_K) containing negatives — which only holds if the signed
-- int8 + correct 292-byte block layout are right. Skips if no model.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Environment_Variables;
with Ada.Unchecked_Deallocation;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;

procedure Test_Weights_Real is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Assert;

   function Model_Path return String is
      Var : constant String := "QWEN_MODEL_PATH";
   begin
      if Ada.Environment_Variables.Exists (Var) then
         return Ada.Environment_Variables.Value (Var);
      end if;
      return "/Users/ceo/.lmstudio/models/HauhauCS/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";
   end Model_Path;

   G : GGUF_File;

   type Raw_Access is access String;
   procedure Free is new Ada.Unchecked_Deallocation (String, Raw_Access);

   function Load_Tensor (Name : String) return Tensor is
      Info : constant Tensor_Info := Find_Tensor (G, Name);
      Size : constant Natural := Natural (Tensor_Byte_Size (Info));
      Raw  : Raw_Access := new String (1 .. Size);
   begin
      Read_Tensor_Raw (G, Info, Raw.all'Address, Size);
      return R : constant Tensor := LLM_Dequant.Dequantize (Info, Raw.all) do
         Free (Raw);
      end return;
   end Load_Tensor;

   --  Inspect a tensor and assert basic sanity (+ negatives for signed quants).
   procedure Check (Name : String; Expect_Neg : Boolean) is
      T        : constant Tensor := Load_Tensor (Name);
      Cnt      : constant Integer := Numel (T);
      Max_Abs  : Float := 0.0;
      Min_V    : Float := Float'Last;
      Max_V    : Float := Float'First;
      Nonzero  : Natural := 0;
      All_Finite : Boolean := True;
      V        : Float;
   begin
      for I in 1 .. Cnt loop
         V := Get_Flat (T, I);
         if not (V = V) or else abs V > Float'Last then
            All_Finite := False;
         else
            if abs V > Max_Abs then Max_Abs := abs V; end if;
            if V < Min_V then Min_V := V; end if;
            if V > Max_V then Max_V := V; end if;
            if V /= 0.0 then Nonzero := Nonzero + 1; end if;
         end if;
      end loop;

      Put_Line ("  " & Name & ": n=" & Integer'Image (Cnt)
                & " min=" & Float'Image (Min_V)
                & " max=" & Float'Image (Max_V)
                & " maxabs=" & Float'Image (Max_Abs));
      Assert (Name & ": all finite", All_Finite);
      Assert (Name & ": has nonzero values", Nonzero > 0);
      Assert (Name & ": weight-magnitude (maxabs<1000)", Max_Abs < 1000.0);
      if Expect_Neg then
         Assert (Name & ": has negatives (signed int8 ok)", Min_V < 0.0);
      end if;
   end Check;

   --  Verify a tensor loads with the expected row-major logical shape.
   procedure Check_Shape (Name : String; E1, E2 : Integer) is
      T : constant Tensor := Load_Tensor (Name);
   begin
      Assert (Name & ": shape [" & Integer'Image (E1) & "," & Integer'Image (E2)
              & " ]",
        Rank (T) = 2 and then Shape (T) (1) = E1 and then Shape (T) (2) = E2);
   end Check_Shape;

begin
   Put_Line ("=== Real-Weight Load/Dequant Test Suite ===");
   New_Line;

   Open (G, Model_Path);
   if not Is_Open (G) or else Tensor_Count (G) = 0 then
      Put_Line ("  SKIP: model not found at " & Model_Path);
      return;
   end if;
   New_Line;

   --  F32 tensors (norms, router).
   Check ("output_norm.weight", Expect_Neg => False);
   Check ("blk.0.attn_norm.weight", Expect_Neg => False);
   Check ("blk.0.ffn_gate_inp.weight", Expect_Neg => True);

   --  Q5_K tensors (the dominant quant type).
   Check ("blk.0.ffn_gate_shexp.weight", Expect_Neg => True);
   Check ("blk.0.attn_gate.weight", Expect_Neg => True);

   --  Q6_K tensors.
   Check ("blk.0.ffn_down_shexp.weight", Expect_Neg => True);
   Check ("blk.0.attn_qkv.weight", Expect_Neg => True);

   --  Shapes are the GGUF dims reversed (row-major logical order).
   New_Line;
   Check_Shape ("blk.0.ffn_gate_inp.weight", 256, 2048);    -- GGUF [2048,256]
   Check_Shape ("blk.0.ffn_down_shexp.weight", 2048, 512);  -- GGUF [512,2048]

   --  3D expert tensor: validate the GGUF dims it will be reversed from.
   declare
      Info : constant Tensor_Info := Find_Tensor (G, "blk.0.ffn_up_exps.weight");
   begin
      Assert ("ffn_up_exps is 3D [dim,ffn,expert] GGUF",
        Natural (Info.N_Dims) = 3
        and then Info.Dims (1) = 2048   -- dim
        and then Info.Dims (2) = 512    -- ffn
        and then Info.Dims (3) = 256);  -- expert
   end;

   Close (G);

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Weights_Real;
