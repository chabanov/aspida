------------------------------------------------------------------------
-- test_q2k_real — validate the Q2_K decoder on a REAL model tensor.
-- Q2_K has no from-scratch quantizer (round-trip would be circular), so we
-- check it against a downloaded TinyLlama-1.1B Q2_K (45 Q2_K + 110 Q3_K + 1
-- Q6_K tensors): the fused/dense QMatVec paths must agree, and the dequantized
-- weights must be finite and weight-sized (a layout/nibble bug yields NaN/Inf
-- or absurd magnitudes). The decoder is additionally confirmed end-to-end by
-- coherent generation (llama_probe emits "Paris" for "The capital of France
-- is"). Skips if the model is absent.
------------------------------------------------------------------------

with Ada.Text_IO;               use Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Unchecked_Deallocation;
with LLM_GGUF;                   use LLM_GGUF;
with LLM_Dequant;
with LLM_Tensor;                 use LLM_Tensor;

procedure Test_Q2K_Real is
   Pass : Boolean := True;
   procedure Assert (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Assert;

   function Model_Path return String is
      Var : constant String := "Q2K_MODEL_PATH";
   begin
      if Ada.Environment_Variables.Exists (Var) then
         return Ada.Environment_Variables.Value (Var);
      end if;
      return "/Users/ceo/.lmstudio/models/TheBloke/"
        & "TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q2_K.gguf";
   end Model_Path;

   G : GGUF_File;

   type Raw_Access is access String;
   procedure Free is new Ada.Unchecked_Deallocation (String, Raw_Access);

   procedure Check (Name : String) is
      Info : constant Tensor_Info := Find_Tensor (G, Name);
      Size : constant Natural := Natural (Tensor_Byte_Size (Info));
      Raw  : Raw_Access := new String (1 .. Size);
      In_Dim  : constant Integer := Integer (Info.Dims (1));
      Out_Dim : constant Integer := Integer (Info.Dims (2));
      X       : Tensor := New_Tensor ([1, In_Dim]);
      Max_Err : Float := 0.0;
      Max_Abs : Float := 0.0;
      Finite  : Boolean := True;
   begin
      Assert (Name & " is Q2_K", Info.Kind = GGML_TYPE_Q2_K);
      Read_Tensor_Raw (G, Info, Raw.all'Address, Size);
      for I in 1 .. In_Dim loop
         Set_Flat (X, I, 0.01 * Float (((I mod 7) - 3)));
      end loop;

      declare
         Y_Q : constant Tensor := LLM_Dequant.QMatVec (Info, Raw.all, X);
         T   : constant Tensor := LLM_Dequant.Dequantize (Info, Raw.all);
      begin
         for O in 1 .. Out_Dim loop
            declare Acc : Float := 0.0; begin
               for I in 1 .. In_Dim loop
                  declare W : constant Float := Get (T, [O, I]); begin
                     if not W'Valid then Finite := False; end if;
                     if abs W > Max_Abs then Max_Abs := abs W; end if;
                     Acc := Acc + W * Get_Flat (X, I);
                  end;
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
                & " max|QMatVec-dense|=" & Float'Image (Max_Err)
                & " max|w|=" & Float'Image (Max_Abs));
      Assert (Name & ": QMatVec matches dense dequant", Max_Err < 1.0e-3);
      Assert (Name & ": weights finite",               Finite);
      Assert (Name & ": weights weight-sized (0<max<8)",
              Max_Abs > 0.0 and then Max_Abs < 8.0);
   end Check;

begin
   Put_Line ("=== Q2_K Real-Tensor Test ===");
   New_Line;
   Open (G, Model_Path);
   if not Is_Open (G) or else Tensor_Count (G) = 0 then
      Put_Line ("  SKIP: model not found at " & Model_Path);
      return;
   end if;

   --  TinyLlama Q2_K stores attn_q/k as Q2_K (the rest is Q3_K/Q6_K).
   Check ("blk.0.attn_q.weight");
   Check ("blk.0.attn_k.weight");
   Close (G);

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Q2K_Real;
