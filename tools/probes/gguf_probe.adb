---------------------------------------------------------------------
-- GGUF_Probe — dump GGUF metadata + key tensor shapes for config audit
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded;
with LLM_GGUF;

procedure GGUF_Probe is
   use Ada.Text_IO;
   use LLM_GGUF;

   File : GGUF_File;
   Path : constant String := Ada.Command_Line.Argument (1);

   function Dims_Img (T : Tensor_Info) return String is
      use Ada.Strings.Unbounded;
      S : Unbounded_String;
   begin
      for D in 1 .. Natural (T.N_Dims) loop
         if D > 1 then
            Append (S, " x ");
         end if;
         Append (S, U64'Image (T.Dims (D)));
      end loop;
      return To_String (S);
   end Dims_Img;

   procedure Dump_Tensor (Name : String) is
   begin
      declare
         T : constant Tensor_Info := Find_Tensor (File, Name);
      begin
         Put_Line ("  " & Name & "  [" & Dims_Img (T) & " ]  "
                   & GGML_Type'Image (T.Kind));
      end;
   exception
      when others =>
         Put_Line ("  " & Name & "  <not found>");
   end Dump_Tensor;

begin
   Put_Line ("=== GGUF Probe ===");
   Put_Line ("File: " & Path);
   Open (File, Path);
   if not Is_Open (File) then
      Put_Line ("Failed to open file.");
      return;
   end if;

   New_Line;
   Put_Line ("Tensors: " & Natural'Image (Tensor_Count (File))
             & "   Metadata: " & Natural'Image (Metadata_Count (File)));
   Put_Line ("Tokenizer tokens: " & Natural'Image (Token_Count (File))
             & "   merges: " & Natural'Image (Merge_Count (File)));

   New_Line;
   Put_Line ("===== ALL METADATA =====");
   for I in 1 .. Metadata_Count (File) loop
      declare
         K  : constant String := Meta_Key_At (File, I);
         V  : constant String := Meta_Value_At (File, I);
         VS : constant String :=
           (if V'Length > 100 then V (V'First .. V'First + 99) & " ..." else V);
      begin
         Put_Line ("  " & K & " = " & VS);
      end;
   end loop;

   New_Line;
   Put_Line ("===== KEY TENSOR SHAPES =====");
   Dump_Tensor ("token_embd.weight");
   Dump_Tensor ("output.weight");
   Dump_Tensor ("output_norm.weight");

   New_Line;
   Put_Line ("--- block 0 ---");
   Dump_Tensor ("blk.0.attn_norm.weight");
   Dump_Tensor ("blk.0.attn_qkv.weight");
   Dump_Tensor ("blk.0.attn_q.weight");
   Dump_Tensor ("blk.0.attn_k.weight");
   Dump_Tensor ("blk.0.attn_v.weight");
   Dump_Tensor ("blk.0.attn_output.weight");
   Dump_Tensor ("blk.0.attn_gate.weight");
   Dump_Tensor ("blk.0.post_attention_norm.weight");
   Dump_Tensor ("blk.0.ffn_gate_inp.weight");
   Dump_Tensor ("blk.0.ffn_gate_exps.weight");
   Dump_Tensor ("blk.0.ffn_up_exps.weight");
   Dump_Tensor ("blk.0.ffn_down_exps.weight");
   Dump_Tensor ("blk.0.ffn_gate_shexp.weight");
   Dump_Tensor ("blk.0.ffn_up_shexp.weight");
   Dump_Tensor ("blk.0.ffn_down_shexp.weight");
   Dump_Tensor ("blk.0.ffn_gate_inp_shexp.weight");

   New_Line;
   Put_Line ("--- block 1 (possible SSM layer) ---");
   Dump_Tensor ("blk.1.attn_norm.weight");
   Dump_Tensor ("blk.1.attn_qkv.weight");
   Dump_Tensor ("blk.1.ssm_conv1d.weight");
   Dump_Tensor ("blk.1.ssm_a");
   Dump_Tensor ("blk.1.ssm_a.weight");
   Dump_Tensor ("blk.1.ssm_dt.weight");
   Dump_Tensor ("blk.1.ssm_norm.weight");
   Dump_Tensor ("blk.1.ssm_out.weight");
   Dump_Tensor ("blk.1.ssm_in.weight");

   New_Line;
   Put_Line ("===== ALL blk.0 / blk.1 TENSORS =====");
   for I in 1 .. Tensor_Count (File) loop
      declare
         T    : constant Tensor_Info := Tensor_At (File, I);
         Name : constant String := Ada.Strings.Unbounded.To_String (T.Name);
      begin
         if Name'Length >= 6
           and then Name (Name'First .. Name'First + 3) = "blk."
           and then Name (Name'First + 4) in '2' .. '4'
           and then Name (Name'First + 5) = '.'
         then
            Put_Line ("  " & Name & "  [" & Dims_Img (T) & " ]  "
                      & GGML_Type'Image (T.Kind));
         end if;
      end;
   end loop;

   New_Line;
   Put_Line ("===== FIRST TOKENS =====");
   for I in 1 .. Integer'Min (8, Token_Count (File)) loop
      Put_Line ("  [" & Integer'Image (I - 1) & "] = '" & Token_At (File, I) & "'");
   end loop;
   if Merge_Count (File) > 0 then
      Put_Line ("  first merge = '" & Merge_At (File, 1) & "'");
   end if;

   Close (File);
   Put_Line ("Done.");
end GGUF_Probe;
