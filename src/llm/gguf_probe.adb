---------------------------------------------------------------------
-- GGUF_Probe — Test program: parse GGUF file, print metadata & tensor info
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_GGUF;

procedure GGUF_Probe is
   use Ada.Text_IO;
   use LLM_GGUF;

   File : GGUF_File;
   Path : constant String := Ada.Command_Line.Argument (1);
begin
   Put_Line ("=== GGUF Probe ===");
   Put_Line ("File: " & Path);

   Open (File, Path);

   if not Is_Open (File) then
      Put_Line ("Failed to open file.");
      return;
   end if;

   Put_Line ("Alignment: " & U64'Image (Alignment (File)));
   Put_Line ("Tensors: " & Natural'Image (Tensor_Count (File)));
   Put_Line ("Metadata: " & Natural'Image (Metadata_Count (File)));
   New_Line;

   -- Print key metadata
   Put_Line ("--- Architecture ---");
   Put_Line ("  arch: " & Metadata (File, "general.architecture"));
   Put_Line ("  name: " & Metadata (File, "general.name"));
   Put_Line ("  context_length: " & Metadata (File, "qwen3.context_length"));
   Put_Line ("  embedding_length: " & Metadata (File, "qwen3.embedding_length"));
   Put_Line ("  block_count: " & Metadata (File, "qwen3.block_count"));
   Put_Line ("  head_count: " & Metadata (File, "qwen3.attention.head_count"));
   Put_Line ("  vocab_size: " & Metadata (File, "qwen3.vocab_size"));
   New_Line;

   -- Print first few tensors
   Put_Line ("--- First 10 Tensors ---");
   for I in 1 .. Integer'Min (10, Tensor_Count (File)) loop
      declare
         T : constant Tensor_Info := Tensor_At (File, I);
         Name : constant String := Ada.Strings.Unbounded.To_String (T.Name);
      begin
         Put_Line (Integer'Image (I) & ". " & Name &
                   " type=" & GGML_Type'Image (T.Kind) &
                   " dims=" & U64'Image (T.Dims (1)) &
                   "x" & U64'Image (T.Dims (2)) &
                   " offset=" & U64'Image (T.Offset));
      end;
   end loop;

   Close (File);
   Put_Line ("Done.");
end GGUF_Probe;
