--  Q4K_Probe — dequantize the first Q4_K super-block of a named tensor and
--  print it, to validate LLM_Dequant.Dequant_Q4_K against a reference (no full
--  model load: only 144 bytes are read from the file).
with Ada.Command_Line;
with Ada.Text_IO;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Tensor;  use LLM_Tensor;
with LLM_Dequant;

procedure Q4K_Probe is
   Path : constant String := Ada.Command_Line.Argument (1);
   Name : constant String :=
     (if Ada.Command_Line.Argument_Count >= 2
      then Ada.Command_Line.Argument (2) else "token_embd.weight");
   G    : GGUF_File;
   Buf  : aliased String (1 .. 144);
   Q    : Tensor := New_Tensor ([1, 256]);
begin
   Open (G, Path);
   declare
      Info : constant Tensor_Info := Find_Tensor (G, Name);
   begin
      Ada.Text_IO.Put_Line ("tensor " & Name & " kind=" & Info.Kind'Image);
      Read_Tensor_Range (G, Info, 0, Buf'Address, 144);
   end;
   Close (G);

   LLM_Dequant.Dequant_Q4_K (Buf, Q, 256);
   Ada.Text_IO.Put ("first8:");
   for I in 1 .. 8 loop Ada.Text_IO.Put (Float'Image (Get_Flat (Q, I))); end loop;
   Ada.Text_IO.New_Line;
   Ada.Text_IO.Put ("last8: ");
   for I in 249 .. 256 loop Ada.Text_IO.Put (Float'Image (Get_Flat (Q, I))); end loop;
   Ada.Text_IO.New_Line;
end Q4K_Probe;
