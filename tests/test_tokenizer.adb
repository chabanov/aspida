---------------------------------------------------------------------
-- Test LLM_Tokenizer — byte-level BPE encode/decode
--
-- Uses a tiny synthetic vocabulary + merge table (no model needed) to
-- verify greedy merging, merge-rank priority, decode round-trip, and the
-- byte-level fallback when no vocabulary is loaded.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tokenizer; use LLM_Tokenizer;

procedure Test_Tokenizer is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Assert;

   function Eq (A, B : Token_Array) return Boolean is
   begin
      if A'Length /= B'Length then
         return False;
      end if;
      for I in 0 .. A'Length - 1 loop
         if A (A'First + I) /= B (B'First + I) then
            return False;
         end if;
      end loop;
      return True;
   end Eq;

begin
   Put_Line ("=== Tokenizer Test Suite ===");
   New_Line;

   ------------------------------------------------------------------
   -- Greedy merge: a,b,c with merges (a b)->ab, (ab c)->abc.
   ------------------------------------------------------------------
   declare
      T : Tokenizer := Create;
   begin
      Add_Token (T, "a", 0);  Add_Token (T, "b", 1);  Add_Token (T, "c", 2);
      Add_Token (T, "ab", 3); Add_Token (T, "abc", 4); Add_Token (T, "bc", 5);
      Add_Merge (T, "a b", 1);
      Add_Merge (T, "ab c", 2);
      Mark_Loaded (T);

      Assert ("vocab size = 6", Vocab_Size (T) = 6);
      Assert ("encode ""abc"" -> [4]", Eq (Encode (T, "abc"), [4]));
      Assert ("encode ""ab""  -> [3]", Eq (Encode (T, "ab"), [3]));
      Assert ("encode ""bca"" -> [1,2,0]", Eq (Encode (T, "bca"), [1, 2, 0]));
      Assert ("decode [4] -> ""abc""", Decode (T, [4]) = "abc");
      Assert ("round-trip ""abc""", Decode (T, Encode (T, "abc")) = "abc");
   end;

   ------------------------------------------------------------------
   -- Rank priority: when two merges are possible, the lower rank wins.
   --   merges: (a b)->rank 2, (b c)->rank 1  => "abc" merges b,c first.
   ------------------------------------------------------------------
   declare
      T : Tokenizer := Create;
   begin
      Add_Token (T, "a", 0); Add_Token (T, "b", 1); Add_Token (T, "c", 2);
      Add_Token (T, "bc", 5);
      Add_Merge (T, "a b", 2);
      Add_Merge (T, "b c", 1);
      Mark_Loaded (T);

      Assert ("lower rank merges first -> [0,5]", Eq (Encode (T, "abc"), [0, 5]));
   end;

   ------------------------------------------------------------------
   -- Byte-level fallback: no vocabulary loaded.
   ------------------------------------------------------------------
   declare
      T : constant Tokenizer := Create;  -- not marked loaded
   begin
      Assert ("not loaded", not Is_Loaded (T));
      Assert ("fallback encode ""Hi"" -> [72,105]",
        Eq (Encode (T, "Hi"), [72, 105]));
      Assert ("fallback round-trip ""Hello""",
        Decode (T, Encode (T, "Hello")) = "Hello");
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Tokenizer;
