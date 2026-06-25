------------------------------------------------------------------------
-- gen_verify — Track 2, Phase 2: a REAL model generates code, a REAL
-- interpreter verifies it. A loaded LLM (via LLM_Llama.Chat) is prompted to
-- write each benchmark function; the model's output is stripped to its code and
-- run by Exec_Verifier (python3) against the spec's tests. Reports the model's
-- (the "teacher") pass-rate and which problems it solved — the verified subset
-- is exactly what a student would distill from. No training here (a large
-- student needs GPU); this closes generate -> verify with a real model.
--
--   QWEN_MODEL_PATH=/path/to/model.gguf  ./obj/gen_verify
------------------------------------------------------------------------

with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Strings.Unbounded;   use Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Environment_Variables;
with Ada.Command_Line;        use Ada.Command_Line;
with LLM_Llama;
with LLM_Qwen;
with LLM_Sampler;
with Exec_Verifier;           use Exec_Verifier;

procedure Gen_Verify is
   Model_Path : constant String :=
     Ada.Environment_Variables.Value ("QWEN_MODEL_PATH", "/root/models/llama8b.gguf");

   --  Keep the code-looking lines from a chat reply (drop prose and ``` fences).
   function Extract_Code (Raw : String) return String is
      Result  : Unbounded_String;
      Started : Boolean := False;
      I       : Integer := Raw'First;

      function Lstrip (S : String) return String is
        (Ada.Strings.Fixed.Trim (S, Ada.Strings.Left));

      function Begins (S, Pfx : String) return Boolean is
        (S'Length >= Pfx'Length
         and then S (S'First .. S'First + Pfx'Length - 1) = Pfx);

      function Is_Code (L : String) return Boolean is
        (L'Length = 0
         or else L (L'First) = ' ' or else L (L'First) = ASCII.HT
         or else Begins (L, "def ") or else Begins (L, "return")
         or else Begins (L, "import ") or else Begins (L, "from ")
         or else (L'Length >= 1 and then L (L'First) = '@'));
   begin
      while I <= Raw'Last loop
         declare
            J : Integer := I;
         begin
            while J <= Raw'Last and then Raw (J) /= ASCII.LF loop J := J + 1; end loop;
            declare
               Line : constant String := Raw (I .. J - 1);
               Trm  : constant String := Lstrip (Line);
            begin
               if Begins (Trm, "```") then
                  null;                                  -- skip markdown fence
               elsif not Started then
                  if Begins (Trm, "def ") then
                     Started := True;
                     Append (Result, Line & ASCII.LF);
                  end if;
               elsif Is_Code (Line) then
                  Append (Result, Line & ASCII.LF);
               else
                  exit;                                  -- first prose line ends it
               end if;
            end;
            I := J + 1;
         end;
      end loop;
      return To_String (Result);
   end Extract_Code;

   Vf      : Python_Verifier;
   Solved  : Natural := 0;
   Params  : constant LLM_Sampler.Params := LLM_Sampler.Greedy;
begin
   Put_Line ("=== gen_verify: real model generates code, real interpreter verifies ===");
   if not Available then
      Put_Line ("SKIP: python3 not found"); return;
   end if;
   Put_Line ("model: " & Model_Path);

   declare
      M : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Model_Path);
   begin
      for P in Problem_Id loop
         declare
            Conv : constant LLM_Qwen.Message_Array :=
              [(LLM_Qwen.Role_System,
                To_Unbounded_String
                  ("You are a Python coder. Output ONLY the function definition,"
                   & " no explanation and no markdown fences.")),
               (LLM_Qwen.Role_User, To_Unbounded_String (Prompt (P)))];
            Reply : constant String :=
              LLM_Llama.Chat (M, Conv, Max_New_Tokens => 64, Params => Params);
            Code  : constant String := Extract_Code (Reply);
            OKr   : constant Boolean := Vf.Is_Correct (P, Code);
         begin
            if OKr then Solved := Solved + 1; end if;
            Put_Line ("  [" & Name (P) & "]  "
                      & (if OKr then "VERIFIED" else "rejected"));
         end;
      end loop;
   end;

   New_Line;
   Put_Line ("model solved (verified by execution):" & Solved'Image
             & " /" & N_Problems'Image);
   if Solved = 0 then
      Put_Line ("RESULT: CHECK  (model produced no verifiable solution)");
      Set_Exit_Status (Failure);
   else
      Put_Line ("RESULT: PASS  (real model -> real interpreter verified "
                & Solved'Image & " solutions; the verified set is what to distill)");
   end if;
end Gen_Verify;
