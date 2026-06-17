--  Meta_Dump — print one metadata value (e.g. tokenizer.chat_template) in
--  full, with no truncation, to stdout. Usage: meta_dump <model.gguf> <key>
with Ada.Command_Line;
with Ada.Text_IO;
with LLM_GGUF; use LLM_GGUF;

procedure Meta_Dump is
   G : GGUF_File;
begin
   Open (G, Ada.Command_Line.Argument (1));
   Ada.Text_IO.Put_Line (Metadata (G, Ada.Command_Line.Argument (2)));
   Close (G);
end Meta_Dump;
