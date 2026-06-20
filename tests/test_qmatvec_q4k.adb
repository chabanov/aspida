---------------------------------------------------------------------
-- Validate the fused Q4_K streaming matrix-vector against full dequant on
-- REAL Q4_K tensors. The bulk Dequant_Q4_K path is left untouched and used
-- as the independent reference: if QMatVec (fused Decode_Q4K_Block) matches
-- the dense matvec of Dequantize, the fused decode+dot is correct.
--
-- Model-agnostic: enumerates tensors, validates the first few 2D Q4_K
-- weights within a size cap. Set ASPIDA_Q4K_MODEL to choose the file; skips
-- (PASS, 0 checks) if no Q4_K model is present.
---------------------------------------------------------------------

with Ada.Text_IO;            use Ada.Text_IO;
with Ada.Command_Line;
with Ada.Environment_Variables;
with Ada.Strings.Unbounded;  use Ada.Strings.Unbounded;
with Ada.Unchecked_Deallocation;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;

procedure Test_QMatVec_Q4K is
   --  GGML_Type equality is predefined; `use LLM_GGUF` above already makes the
   --  type visible, so no `use type` is needed here.

   Passed, Failed, Checked : Natural := 0;
   Max_Checks : constant := 4;          -- enough to be convincing, stays fast
   Dim_Cap    : constant := 4096;       -- bound dense-dequant cost

   function Model_Path return String is
      Var : constant String := "ASPIDA_Q4K_MODEL";
   begin
      if Ada.Environment_Variables.Exists (Var) then
         return Ada.Environment_Variables.Value (Var);
      end if;
      return "/Users/ceo/.lmstudio/models/HauhauCS/"
        & "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/"
        & "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf";
   end Model_Path;

   G : GGUF_File;

   type Raw_Access is access String;
   procedure Free is new Ada.Unchecked_Deallocation (String, Raw_Access);

   --  Compare fused QMatVec to the dense (full-dequant) matvec on one tensor.
   procedure Check (Info : Tensor_Info) is
      Name    : constant String  := To_String (Info.Name);
      Size    : constant Natural := Natural (Tensor_Byte_Size (Info));
      Raw     : Raw_Access := new String (1 .. Size);
      In_Dim  : constant Integer := Integer (Info.Dims (1));
      Out_Dim : constant Integer := Integer (Info.Dims (2));
      X       : Tensor  := New_Tensor ([1, In_Dim]);
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
      Checked := Checked + 1;
      if Max_Err < 1.0e-3 then
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Check;

begin
   Put_Line ("=== Fused Q4_K MatVec Real-Tensor Test Suite ===");
   New_Line;

   Open (G, Model_Path);
   if not Is_Open (G) or else Tensor_Count (G) = 0 then
      Put_Line ("  SKIP: Q4_K model not found at " & Model_Path);
      return;
   end if;

   for I in 1 .. Tensor_Count (G) loop
      exit when Checked >= Max_Checks;
      declare
         Info : constant Tensor_Info := Tensor_At (G, I);
      begin
         if Info.Kind = LLM_GGUF.GGML_TYPE_Q4_K
           and then Info.N_Dims = 2
           and then Integer (Info.Dims (1)) mod 256 = 0
           and then Integer (Info.Dims (1)) in 1 .. Dim_Cap
           and then Integer (Info.Dims (2)) in 1 .. Dim_Cap
         then
            Check (Info);
         end if;
      end;
   end loop;

   New_Line;
   if Checked = 0 then
      Put_Line ("  SKIP: no in-range Q4_K 2D tensor found in this model");
   end if;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image
             & " failed," & Checked'Image & " checked.");
   Close (G);
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_QMatVec_Q4K;
