--  quant_probe — dequantize the first 32-element block of a named tensor via
--  the real LLM_Dequant dispatch (whatever its GGML kind) and print it, to
--  validate Q4_0 / Q5_0 (and others) against a reference. No full model load.
with Ada.Command_Line;
with Ada.Text_IO;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Tensor;  use LLM_Tensor;
with LLM_Dequant;

procedure Quant_Probe is
   Path : constant String := Ada.Command_Line.Argument (1);
   Name : constant String := Ada.Command_Line.Argument (2);
   G    : GGUF_File;
begin
   Open (G, Path);
   declare
      Info : constant Tensor_Info := Find_Tensor (G, Name);
      Blk  : Tensor_Info := Info;        -- a one-block (32-elem) view
   begin
      Blk.N_Dims := 1;
      Blk.Dims   := [32, 0, 0, 0];
      declare
         NB  : constant Natural := Natural (Tensor_Byte_Size (Blk));  -- bytes/block
         Buf : aliased String (1 .. NB);
         Q   : Tensor;
      begin
         Read_Tensor_Range (G, Info, 0, Buf'Address, NB);
         Q := LLM_Dequant.Dequantize (Blk, Buf);
         Ada.Text_IO.Put_Line (Name & " kind=" & Info.Kind'Image
           & " block_bytes=" & NB'Image);
         Ada.Text_IO.Put ("vals:");
         for I in 1 .. 32 loop
            Ada.Text_IO.Put (Float'Image (Get_Flat (Q, I)));
         end loop;
         Ada.Text_IO.New_Line;
      end;
   end;
   Close (G);
end Quant_Probe;
