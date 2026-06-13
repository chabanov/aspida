---------------------------------------------------------------------
-- LLM_Chat body — interactive chat loop
-- Supports Qwen 3.5 GGUF, GPT-2 weights, and tiny random models
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Exceptions;
with LLM_Model;
with LLM_Qwen;

package body LLM_Chat is

   procedure Run (Model_Dim : Integer := 64; N_Layers : Integer := 2) is
      use Ada.Text_IO;

      -- Qwen model (from GGUF)
      Qwen_M : LLM_Qwen.Qwen_Model;
      -- GPT-2 model (from our converter)
      GPT2_M : LLM_Model.GPT_Model;
      -- Which backend is active
      Use_Qwen : Boolean := False;
      Use_GPT2 : Boolean := False;

      Params : Long_Long_Integer := 0;
      Buffer : String (1 .. 4096);
      Last   : Integer;

      QWEN_PATH : constant String :=
        "/Users/ceo/.lmstudio/models/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";
   begin
      -- Try Qwen 3.5 first
      begin
         Qwen_M := LLM_Qwen.Load (QWEN_PATH);
         Put_Line ("=== Aspida LLM Chat (Qwen 3.5 MoE) ===");
         Use_Qwen := True;
         Params := LLM_Qwen.Param_Count (Qwen_M);
      exception
         when E : others =>
            Put_Line ("Qwen load failed: " & Ada.Exceptions.Exception_Message (E));
            Put_Line ("Trying GPT-2 fallback...");
            -- Try GPT-2 next
            begin
               GPT2_M := LLM_Model.Load_GPT2 ("models/gpt2_small");
               Put_Line ("=== Aspida LLM Chat (GPT-2 Small) ===");
               Use_GPT2 := True;
               Params := Long_Long_Integer (LLM_Model.Param_Count (GPT2_M));
            exception
               when others =>
                  Put_Line ("GPT-2 not found, using random tiny model.");
                  GPT2_M := LLM_Model.New_Tiny (Model_Dim, N_Layers);
                  Use_GPT2 := True;
                  Params := Long_Long_Integer (LLM_Model.Param_Count (GPT2_M));
            end;
      end;

      Put_Line ("Params: " & Long_Long_Integer'Image (Params));
      Put_Line ("");
      Put_Line ("Commands: /quit, /clear, /model");
      Put_Line ("--------------------------");

      loop
         Put ("> ");
         Flush;
         Get_Line (Buffer, Last);

         if Last = 0 then
            goto Continue;
         end if;

         declare
            Input : constant String := Buffer (1 .. Last);
         begin
            if Input = "/quit" then
               Put_Line ("Bye.");
               exit;

            elsif Input = "/clear" then
               Put_Line ("Context cleared.");

            elsif Input = "/model" then
               if Use_Qwen then
                  Put_Line ("Qwen 3.5 MoE | " &
                    Integer'Image (LLM_Qwen.Block_Count (Qwen_M)) & " blocks, " &
                    Integer'Image (LLM_Qwen.Dim (Qwen_M)) & "d, " &
                    Long_Long_Integer'Image (Params) & " params");
               elsif Use_GPT2 then
                  Put_Line ("GPT-2 Small | " &
                    Integer'Image (LLM_Model.Param_Count (GPT2_M)) & " params");
               else
                  Put_Line ("Tiny random model");
               end if;

            else
               declare
                  Response : String (1 .. 4096);
                  Resp_Len : Integer := 0;
               begin
                  if Use_Qwen then
                     declare
                        R : constant String := LLM_Qwen.Generate (Qwen_M, Input, 100);
                     begin
                        Response (1 .. R'Length) := R;
                        Resp_Len := R'Length;
                     end;
                  else
                     declare
                        R : constant String := LLM_Model.Generate (GPT2_M, Input, 100);
                     begin
                        Response (1 .. R'Length) := R;
                        Resp_Len := R'Length;
                     end;
                  end if;
                  Put_Line (Response (1 .. Resp_Len));
               end;
            end if;
         end;

      <<Continue>>
         null;
      end loop;
   end Run;

end LLM_Chat;
