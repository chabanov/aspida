---------------------------------------------------------------------
-- Validate streaming quantized matrix-vector against full dequant, on
-- REAL model tensors (Q5_K and Q6_K). QMatVec must equal the dense
-- matvec of the fully dequantized weight while only ever holding one
-- row of FP32 at a time. Skips if the model is absent.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Environment_Variables;
with Ada.Unchecked_Deallocation;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;

procedure Test_QMatVec_Real is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);  Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);  Failed := Failed + 1;
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

   --  Compare QMatVec(W, x) to dense (full-dequant) matvec on a named tensor.
   procedure Check (Name : String) is
      Info : constant Tensor_Info := Find_Tensor (G, Name);
      Size : constant Natural := Natural (Tensor_Byte_Size (Info));
      Raw  : Raw_Access := new String (1 .. Size);
      In_Dim  : constant Integer := Integer (Info.Dims (1));
      Out_Dim : constant Integer := Integer (Info.Dims (2));
      X       : Tensor := New_Tensor ([1, In_Dim]);
      Max_Err : Float := 0.0;
   begin
      Read_Tensor_Raw (G, Info, Raw.all'Address, Size);
      for I in 1 .. In_Dim loop
         Set_Flat (X, I, 0.01 * Float (((I mod 7) - 3)));
      end loop;

      declare
         Y_Q : constant Tensor := LLM_Dequant.QMatVec (Info, Raw.all, X);
         T   : constant Tensor := LLM_Dequant.Dequantize (Info, Raw.all);  -- [out,in]
      begin
         for O in 1 .. Out_Dim loop
            declare
               Acc : Float := 0.0;
            begin
               for I in 1 .. In_Dim loop
                  Acc := Acc + Get (T, [O, I]) * Get_Flat (X, I);
               end loop;
               if abs (Acc - Get_Flat (Y_Q, O)) > Max_Err then
                  Max_Err := abs (Acc - Get_Flat (Y_Q, O));
               end if;
            end;
         end loop;
      end;
      Free (Raw);

      Put_Line ("  " & Name & "  in=" & Integer'Image (In_Dim)
                & " out=" & Integer'Image (Out_Dim)
                & " max|QMatVec-dense|=" & Float'Image (Max_Err));
      Assert (Name & ": QMatVec matches dense dequant", Max_Err < 1.0e-3);
   end Check;

begin
   Put_Line ("=== Quantized MatVec Real-Tensor Test Suite ===");
   New_Line;

   Open (G, Model_Path);
   if not Is_Open (G) or else Tensor_Count (G) = 0 then
      Put_Line ("  SKIP: model not found at " & Model_Path);
      return;
   end if;
   New_Line;

   Check ("blk.0.ffn_gate_shexp.weight");   -- Q5_K, in=2048 out=512
   Check ("blk.0.ffn_down_shexp.weight");   -- Q6_K, in=512  out=2048

   Close (G);

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_QMatVec_Real;
