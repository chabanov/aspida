------------------------------------------------------------------------
-- test_gguf — write a GGUF with GGUF_Write, then load it with the engine's
-- own reader (LLM_GGUF) and verify metadata, the tokenizer string-array, and
-- F32 tensor data survive byte-for-byte. Proves we can emit engine-loadable
-- model files (the export half of teach->train->serve).
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GGUF_Write;
with LLM_GGUF;

procedure Test_GGUF is
   Pass : Boolean := True;
   Path : constant String := "/tmp/aspida_export_test.gguf";

   procedure Check (Cond : Boolean; Name : String) is
   begin
      Put_Line ("  " & Name & ": " & (if Cond then "OK" else "FAIL"));
      if not Cond then Pass := False; end if;
   end Check;

   --  reference tensor data
   Emb  : constant GGUF_Write.Float_Array :=
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];   -- [3 x 4]
   Outw : constant GGUF_Write.Float_Array :=
     [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 4.5, -4.5, 5.5, -5.5];

begin
   Put_Line ("=== Aspida GGUF write/read round-trip ===");

   --  ---- write ----
   declare
      B    : GGUF_Write.Builder;
      Toks : constant GGUF_Write.Str_List :=
        [To_Unbounded_String ("<bos>"),
         To_Unbounded_String ("hello"),
         To_Unbounded_String ("world")];
   begin
      GGUF_Write.Meta_Str (B, "general.architecture", "llama");
      GGUF_Write.Meta_Str (B, "general.name", "aspida-student-tiny");
      GGUF_Write.Meta_U32 (B, "llama.block_count", 1);
      GGUF_Write.Meta_F32 (B, "llama.attention.layer_norm_rms_epsilon", 1.0E-5);
      GGUF_Write.Meta_Str_Array (B, "tokenizer.ggml.tokens", Toks);
      GGUF_Write.Add_Tensor_F32 (B, "token_embd.weight", [3, 4], Emb);
      GGUF_Write.Add_Tensor_F32 (B, "output.weight",     [4, 3], Outw);
      GGUF_Write.Save (B, Path);
   end;

   --  ---- read back with the engine's own parser ----
   declare
      G : LLM_GGUF.GGUF_File;
   begin
      LLM_GGUF.Open (G, Path);
      Check (LLM_GGUF.Is_Open (G), "file opens as valid GGUF");
      Check (LLM_GGUF.Metadata (G, "general.architecture") = "llama",
             "architecture = llama");
      Check (LLM_GGUF.Metadata (G, "general.name") = "aspida-student-tiny",
             "name metadata");
      Check (LLM_GGUF.Token_Count (G) = 3, "tokenizer has 3 tokens");
      Check (LLM_GGUF.Token_At (G, 1) = "<bos>"
             and then LLM_GGUF.Token_At (G, 2) = "hello"
             and then LLM_GGUF.Token_At (G, 3) = "world",
             "token strings round-trip");

      --  tensor data
      declare
         Info : constant LLM_GGUF.Tensor_Info :=
           LLM_GGUF.Find_Tensor (G, "token_embd.weight");
         N    : constant Natural := Natural (LLM_GGUF.Tensor_Num_Elements (Info));
         Buf  : GGUF_Write.Float_Array (1 .. N);
         OK   : Boolean := (N = Emb'Length);
      begin
         if OK then
            LLM_GGUF.Read_Tensor_Raw (G, Info, Buf'Address, N * 4);
            for I in 1 .. N loop
               if Buf (I) /= Emb (I) then OK := False; end if;
            end loop;
         end if;
         Check (OK, "token_embd.weight data matches (" & N'Image & " floats)");
      end;

      declare
         Info : constant LLM_GGUF.Tensor_Info :=
           LLM_GGUF.Find_Tensor (G, "output.weight");
         N    : constant Natural := Natural (LLM_GGUF.Tensor_Num_Elements (Info));
         Buf  : GGUF_Write.Float_Array (1 .. N);
         OK   : Boolean := (N = Outw'Length);
      begin
         if OK then
            LLM_GGUF.Read_Tensor_Raw (G, Info, Buf'Address, N * 4);
            for I in 1 .. N loop
               if Buf (I) /= Outw (I) then OK := False; end if;
            end loop;
         end if;
         Check (OK, "output.weight data matches");
      end;

      LLM_GGUF.Close (G);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_GGUF;
