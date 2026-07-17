---------------------------------------------------------------------
-- LLM_Chat_Parser body — see spec.
--
-- Approach: each Feed(Piece) call appends to a rolling buffer, then loops
-- a small dispatch that scans for the next chat-template tag (think_open,
-- think_close, tool_open, tool_close) and routes the preceding text to the
-- right sink callback. A safety holdback at the end of the buffer keeps
-- tag-match bytes intact across Feed boundaries. After Finalize() the
-- remaining buffer is flushed and On_Finish_Reason fires.
---------------------------------------------------------------------

with Ada.Strings;        use Ada.Strings;
with Ada.Strings.Fixed;  use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body LLM_Chat_Parser is

   --  Tag literals. We accept BOTH the canonical ChatML/Qwen3.5 form
   --  (angle-bracketed, e.g. "<think>...</think>") AND the abbreviated
   --  "bare" form some fine-tunes emit — including DeepReinforce's
   --  Hura (Qwen3.5 9B). The bare form uses the same token for
   --  open and close (e.g. "think"/"think", "tool_call"/"tool_call"),
   --  so we resolve the ambiguity with a balance counter: the FIRST
   --  occurrence inside a region is the opener, the SECOND closes it.
   --  Canonical "</tool_call>" is always unambiguous (matches only as
   --  closer when inside a tool block).
   Think_Open_Canon  : constant String := "<think>";
   Think_Close_Canon : constant String := "</think>";
   Think_Bare        : constant String := "think";
   Tool_Open_Canon   : constant String := "<tool_call>";
   Tool_Close_Canon  : constant String := "</tool_call>";
   Tool_Bare         : constant String := "tool_call";

   --  Find the earliest match of either canonical or bare form of the
   --  given pair, setting Best and Len accordingly. If neither matches,
   --  Best = 0 and Len = 0. Critically, a bare match is REJECTED when it
   --  is preceded by '<': the model might be in the middle of emitting a
   --  canonical "<think>" or "<tool_call>" tag that hasn't fully arrived
   --  yet, and we must wait for it. This avoids the char-by-char race
   --  where "<t" or "<too" would otherwise false-positive.
   procedure Earliest (Canon : String; Bare : String; Str : String;
                       From : Positive; Bare_Line_Start_Only : Boolean;
                       Best : in out Natural;
                       Len  : in out Natural) is
      Bc : constant Natural := Index (Str, Canon, From);
      Lc : constant Natural := (if Bc > 0 then Canon'Length else 0);
      --  Find the earliest bare tag that satisfies the position rules.
      --  We may need to skip false-positive occurrences (e.g. the word
      --  "think" inside reasoning prose when looking for the bare
      --  closer); keep searching beyond them.
      Bb : Natural := 0;
      Lb : Natural := 0;
      Cursor : Natural := From;
   begin
      Best := Bc;
      Len  := Lc;
      loop
         Bb := Index (Str, Bare, Cursor);
         exit when Bb = 0;
         if Bb > Str'First and then Str (Bb - 1) = '<' then
            --  "<think" or "<tool_call" — partial canonical opener; wait.
            Cursor := Bb + Bare'Length;
         elsif Bb > Str'First
           and then Str'First < Bb - 1
           and then Str (Bb - 1) = '/'
         then
            --  "</think" — partial canonical closer; wait for ">".
            Cursor := Bb + Bare'Length;
         elsif Bare_Line_Start_Only
           and then Bb > Str'First
           and then Str (Bb - 1) /= ASCII.LF
           and then Str (Bb - 1) /= '>'
         then
            --  Bare tag in prose ("I think…", or a parameter value that
            --  happens to contain "tool_call"). Only honour bare tags at
            --  start-of-string or right after a newline / body closer.
            Cursor := Bb + Bare'Length;
         else
            Lb := Bare'Length;
            exit;
         end if;
      end loop;
      if Bb > 0 and then (Best = 0 or else Bb < Best) then
         Best := Bb;
         Len  := Lb;
      end if;
   end Earliest;

   --  Tag literals used at the parse level (canonical form, kept for
   --  Step() internal length lookups). The Step function consults
   --  Index for both forms and remembers which one matched.

   --  Cap on the rolling buffer: we never hold back more than this many
   --  characters past a tag boundary. Pieces longer than this will still
   --  be processed, the inner Step loop just iterates until drained.
   Safety : constant Natural := 32;

   function Idx (Pat : String; Text : String; From : Positive) return Natural is
     (Index (Text, Pat, From));

   function Id_For (N : Natural) return String is
     ("tc_" & Ada.Strings.Fixed.Trim (Natural'Image (N), Ada.Strings.Both));

   function New_Parser (Start_In_Reasoning : Boolean := False) return Parser is
      P : Parser;
   begin
      P.Max_Buf := Safety;
      if Start_In_Reasoning then
         P.State       := S_In_Reasoning;
         P.Think_Depth := 1;
      end if;
      return P;
   end New_Parser;

   --  Quote a string value into a valid JSON literal. Handles the four
   --  characters that MUST be escaped (per RFC 8259); anything else
   --  passes through (we don't strip control bytes — those would make the
   --  parser reject them upstream).
   function Quote (S : String) return String is
      Acc : Unbounded_String := Null_Unbounded_String;
   begin
      Acc := Acc & '"';
      for I in S'Range loop
         case S (I) is
            when '"'  => Acc := Acc & '\' & '"';
            when '\'  => Acc := Acc & '\' & '\';
            when ASCII.LF => Acc := Acc & '\' & 'n';
            when ASCII.CR => Acc := Acc & '\' & 'r';
            when ASCII.HT => Acc := Acc & '\' & 't';
            when others => Acc := Acc & S (I);
         end case;
      end loop;
      Acc := Acc & '"';
      return To_String (Acc);
   end Quote;

   --  Strip leading/trailing whitespace from a parameter key or value. The
   --  model routinely writes `<parameter=limit>\n100\n</parameter>` (value on
   --  its own line), and without this the value came through as "\n100" — a
   --  string the downstream JSON/int coercion rejects, so the tool call errored
   --  ("limit":"\n100"). Interior whitespace is preserved.
   function Trim_WS (S : String) return String is
      function Is_WS (C : Character) return Boolean is
        (C = ' ' or else C = ASCII.LF or else C = ASCII.CR or else C = ASCII.HT);
      F : Integer := S'First;
      L : Integer := S'Last;
   begin
      while F <= L and then Is_WS (S (F)) loop F := F + 1; end loop;
      while L >= F and then Is_WS (S (L)) loop L := L - 1; end loop;
      return S (F .. L);
   end Trim_WS;

   --  Reconstruct a compact JSON object {"key":...,"key2":...} from the
   --  raw <parameter=KEY>VALUE</parameter><parameter=KEY2>VALUE2</parameter>
   --  pairs the model writes inside a tool block. Pairs are emitted in the
   --  order they first appear (which is what OpenAI's stream.deltas carry
   --  anyway). On malformed input we return the empty object so the caller
   --  still gets a parsable (if useless) tool_call.
   function Args_Of (Blk : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Pos    : Natural := Blk'First;
      First  : Boolean := True;
   begin
      loop
         declare
            PS : constant Natural := Idx ("<parameter=", Blk, Pos);
         begin
            exit when PS = 0;
            declare
               KS  : constant Natural := PS + 11;
               KE  : Natural := KS;
               VES : Natural := 0;
               Tail : Natural := 0;
            begin
               while KE <= Blk'Last and then Blk (KE) /= '>'
                       and then (KE - KS) <= 64 loop
                  KE := KE + 1;
               end loop;
               --  position right after ">": the value
               exit when KE > Blk'Last or else Blk (KE) /= '>';
               Tail := KE + 1;
               --  Now scan to </parameter>: skip plain content. We avoid
               --  doing nested tag matching here (parameters are flat), so
               --  </parameter> is searched greedily.
               declare
                  QE : constant Natural :=
                    Idx ("</parameter>", Blk, Tail);
               begin
                  VES := QE;
               end;
               --  No closing </parameter> (a truncated / still-streaming tool
               --  call — e.g. the model was cut off mid-argument). STOP: emit
               --  what parsed so far. Falling through to `Pos := VES + 1` with
               --  VES = 0 rewinds Pos to Blk'First, so the same <parameter= is
               --  re-found every iteration — an INFINITE LOOP that spins a core,
               --  never releases the generation's batch lane, and after 8 such
               --  hangs wedges the whole server (2026-07-14).
               exit when VES = 0;
               declare
                  Key : constant String := Trim_WS (Blk (KS .. KE - 1));
                  Val : constant String := Trim_WS (Blk (Tail .. VES - 1));
               begin
                  if not First then
                     Result := Result & ",";
                  end if;
                  First := False;
                  Result := Result & Quote (Key) & ": " & Quote (Val);
               end;
               Pos := VES + 1;
            end;
         end;
      end loop;
      if First then
         return "{}";
      end if;
      return To_String ("{" & Result & "}");
   end Args_Of;

   --  Extract a <function=NAME>...</function> block from Blk and emit it as a
   --  tool call on Sink. Returns True iff a well-formed call was emitted.
   --  Shared by Step (on the canonical </tool_call> closer) and Finalize —
   --  the Hura fine-tune frequently ends the turn right after </function>,
   --  omitting the outer </tool_call>, so without a Finalize fallback the whole
   --  tool call was buffered and silently dropped (streaming emitted nothing →
   --  the platform reported "empty response / no tool support").
   function Emit_Tool_Block (P    : in out Parser;
                             Blk  : String;
                             Sink : access LLM_Qwen.Chat_Sink'Class)
                             return Boolean
   is
      Fn : constant Natural := Idx ("<function=", Blk, Blk'First);
      Nm : Unbounded_String := Null_Unbounded_String;
      Ar : Unbounded_String := Null_Unbounded_String;
   begin
      if Fn > 0 then
         declare
            NS : constant Natural := Fn + 10;
            NE : Natural := NS;
         begin
            while NE <= Blk'Last and then Blk (NE) /= '>'
                    and then (NE - NS) <= 64 loop
               NE := NE + 1;
            end loop;
            if NE <= Blk'Last and then Blk (NE) = '>' then
               Nm := To_Unbounded_String (Blk (NS .. NE - 1));
               Ar := To_Unbounded_String (Args_Of (Blk));
            end if;
         end;
      end if;

      if not P.Tool_Cap_Reached and then Nm /= Null_Unbounded_String then
         --  Check the cap BEFORE incrementing: N_Calls indexes Calls and is
         --  also the slice bound in Tool_Calls_Of, so letting it reach
         --  Calls'Last + 1 raises Constraint_Error out of every later Chat.
         if P.N_Calls < P.Calls'Last then
            P.N_Calls := P.N_Calls + 1;
            declare
               Idx_C : constant String := Id_For (P.N_Calls);
            begin
               P.Calls (P.N_Calls) :=
                 (Id           => To_Unbounded_String (Idx_C),
                  Name         => Nm,
                  Arguments_JS => Ar);
               LLM_Qwen.On_Tool_Call
                 (Sink.all, Idx_C, To_String (Nm), To_String (Ar));
               P.Text_After_Last_Tool := False;
            end;
            return True;
         else
            P.Tool_Cap_Reached := True;
         end if;
      end if;
      return False;
   end Emit_Tool_Block;

   procedure Emit_Text (P : in out Parser;
                        Text : String;
                        Sink : access LLM_Qwen.Chat_Sink'Class) is
   begin
      if Text'Length = 0 then return; end if;
      case P.State is
         when S_Idle | S_In_Text =>
            P.Answer := P.Answer & Text;
            LLM_Qwen.On_Text (Sink.all, Text);
            if P.State = S_In_Text then
               P.Text_After_Last_Tool := True;
            end if;
         when S_In_Reasoning =>
            P.Reasoning := P.Reasoning & Text;
            LLM_Qwen.On_Reasoning (Sink.all, Text);
         when S_In_Tool =>
            --  Text inside an unclosed tool block falls through to the
            --  answer (the model forgot ② or emitted malformed XML —
            --  losing the text would be worse than mis-classifying it).
            P.Answer := P.Answer & Text;
            LLM_Qwen.On_Text (Sink.all, Text);
      end case;
   end Emit_Text;

   --  Examine Buf and consume one chunk of progress (a tag + its preceding
   --  text). Returns True when something was consumed; False when no more
   --  tags are visible (caller may want to flush safety-holdback text).
   function Step (P : in out Parser;
                  Sink : access LLM_Qwen.Chat_Sink'Class) return Boolean is
      Str : constant String := To_String (P.Buf);
      Len : constant Natural := Length (P.Buf);

      Best  : Natural := 0;
      Len_B : Natural := 0;
      Match : Natural := 0;
   begin
      case P.State is
         when S_Idle =>
            --  First opener wins. Both canonical and bare forms are
            --  accepted; we just need the earliest hit. Line-start rule
            --  is OFF here because the model legitimately starts a chat
            --  reply with a bare tag at Str'First (no preceding char).
            declare
               Th_Op : Natural := 0;
               Th_Len : Natural := 0;
               Tb_Op : Natural := 0;
               Tb_Len : Natural := 0;
            begin
               Earliest (Think_Open_Canon, Think_Bare,
                         Str, Str'First, False, Th_Op, Th_Len);
               Earliest (Tool_Open_Canon, Tool_Bare,
                         Str, Str'First, False, Tb_Op, Tb_Len);
               if Th_Op > 0
                 and then (Tb_Op = 0 or else Th_Op <= Tb_Op)
               then
                  Best := Th_Op; Len_B := Th_Len; Match := 1;
               elsif Tb_Op > 0 then
                  Best := Tb_Op; Len_B := Tb_Len; Match := 2;
               end if;
            end;
         when S_In_Reasoning =>
            --  Bare closer only at line start: reasoning prose may contain
            --  the word "think" as natural language.
            declare
               Cl : Natural := 0;
               Ll : Natural := 0;
            begin
               Earliest (Think_Close_Canon, Think_Bare,
                         Str, Str'First, True, Cl, Ll);
               if Cl > 0 then
                  Best := Cl; Len_B := Ll; Match := 1;
               end if;
            end;
         when S_In_Text =>
            --  New tool_call after text or after a closed reasoning/tool
            --  block. Line-start rule OFF: model may emit the bare opener
            --  at Str'First (e.g. immediately after reasoning closed).
            declare
               Tb : Natural := 0;
               Tl : Natural := 0;
            begin
               Earliest (Tool_Open_Canon, Tool_Bare,
                         Str, Str'First, False, Tb, Tl);
               if Tb > 0 then
                  Best := Tb; Len_B := Tl; Match := 2;
               end if;
            end;
         when S_In_Tool =>
            --  Inside the tool block, parameter values can contain any
            --  text including the literal word "tool_call". Bare closer
            --  only at line start (or Str'First).
            declare
               Open_Pos : Natural := 0;
               Open_Len : Natural := 0;
               Close_Pos : Natural := 0;
               Close_Len : Natural := 0;
            begin
               Earliest (Tool_Open_Canon, Tool_Bare,
                         Str, Str'First, True, Open_Pos, Open_Len);
               Earliest (Tool_Close_Canon, Tool_Bare,
                         Str, Str'First, True, Close_Pos, Close_Len);
               if Open_Pos > 0
                 and then (Close_Pos = 0 or else Open_Pos < Close_Pos)
               then
                  Best := Open_Pos; Len_B := Open_Len; Match := 2;
               elsif Close_Pos > 0 then
                  Best := Close_Pos; Len_B := Close_Len; Match := 3;
               end if;
            end;
      end case;

      if Best = 0 then
         --  No tag in the buffer. Previously we held ALL text until a tag
         --  (or Finalize) appeared, so a whole reasoning block (until
         --  </think>) and the whole answer (until end) each arrived at once
         --  — the client saw the reply "as one message" instead of streaming.
         --  In the text-streaming states, flush everything except a short
         --  tail that could still be the start of a tag spanning the next
         --  Feed. Tag_Tail (24) safely exceeds the longest tag plus a leading
         --  newline (e.g. "\n</tool_call>"), so a partial tag — and the
         --  newline a bare line-start tag needs — always stays buffered and
         --  is matched once completed. Text now streams token-by-token.
         declare
            Tag_Tail : constant := 24;
            Cut      : Integer := Len - Tag_Tail;
         begin
            --  Never split a multi-byte UTF-8 sequence at the cut: back off
            --  any continuation byte (10xxxxxx) so we emit whole characters
            --  and the tail starts on a lead byte. Splitting mid-codepoint
            --  produced replacement-char mojibake (e.g. emoji shown as ���).
            while Cut > 0
              and then Character'Pos (Str (Cut + 1)) in 16#80# .. 16#BF#
            loop
               Cut := Cut - 1;
            end loop;
            if (P.State = S_In_Reasoning or else P.State = S_In_Text
                or else P.State = S_Idle)
              and then Cut > 0
            then
               Emit_Text (P, Str (1 .. Cut), Sink);
               P.Buf := To_Unbounded_String (Str (Cut + 1 .. Len));
            end if;
         end;
         return False;
      end if;

      --  Emit any preceding text, but NOT while we're inside a tool block:
      --  those bytes belong to the parameter list and must reach the
      --  closer-handler verbatim (Blk is captured at tag-close).
      --  Best > 1 (Str'First is always 1 after To_String): emit any
      --  text from Buf'First up to (Best-1).
      if Best > 1 and then P.State /= S_In_Tool then
         Emit_Text (P, Str (1 .. Best - 1), Sink);
      end if;

      --  Consume the matched tag.
      declare
         After : constant Natural := Integer'Min (Best + Len_B, Len + 1);
         Canon_Match : constant Boolean :=
           (Best > 0) and then
           (Len_B = Think_Open_Canon'Length
            or Len_B = Think_Close_Canon'Length
            or Len_B = Tool_Open_Canon'Length
            or Len_B = Tool_Close_Canon'Length);
      begin
         case Match is
            when 1 =>
               if P.State = S_Idle then
                  P.State := S_In_Reasoning;
                  P.Think_Depth := 1;
               else
                  --  S_In_Reasoning: must be a closer.
                  if Canon_Match and then Len_B = Think_Close_Canon'Length
                  then
                     P.Think_Depth := 0;
                  elsif not Canon_Match
                    and then P.Think_Depth > 0
                  then
                     --  Bare "think" closer (first matching): exit.
                     P.Think_Depth := 0;
                  else
                     --  Bare "think" without an opener in scope (depth=0)
                     --  OR a canonical " THINKopen" inside reasoning
                     --  (depth > 0): keep current state, just skip the
                     --  match bytes so they don't get re-matched.
                     if After <= Len then
                        P.Buf := To_Unbounded_String (Str (After .. Len));
                     else
                        P.Buf := Null_Unbounded_String;
                     end if;
                     return True;
                  end if;
                  P.State := S_In_Text;
               end if;
            when 2 =>
               if P.State = S_Idle or else P.State = S_In_Text then
                  P.Tool_Depth := 1;
                  P.State := S_In_Tool;
               else
                  --  Already S_In_Tool: another bare opener. Increment
                  --  balance; we'll look for the matching closer.
                  P.Tool_Depth := P.Tool_Depth + 1;
                  P.State := S_In_Tool;
               end if;
            when 3 =>
               --  Assemble and emit one tool call from the block preceding
               --  this closer.
               declare
                  Emitted : constant Boolean :=
                    Emit_Tool_Block (P, Str (Str'First .. Best - 1), Sink);
               begin
                  pragma Unreferenced (Emitted);
               end;
               P.Tool_Depth := P.Tool_Depth - 1;
               if P.Tool_Depth > 0 then
                  --  Nested: not yet at the outermost close. Stay in
                  --  S_In_Tool; just consume the bytes so the next
                  --  iteration looks for the actual outer close.
                  if After <= Len then
                     P.Buf := To_Unbounded_String (Str (After .. Len));
                  else
                     P.Buf := Null_Unbounded_String;
                  end if;
                  return True;
               end if;
               P.State := S_In_Text;
            when others =>
               null;
         end case;
         if After <= Len then
            P.Buf := To_Unbounded_String (Str (After .. Len));
         else
            P.Buf := Null_Unbounded_String;
         end if;
         return True;
      end;
   end Step;

   procedure Feed (P : in out Parser; Piece : String;
                   Sink : access LLM_Qwen.Chat_Sink'Class) is
   begin
      if Piece'Length = 0 then return; end if;
      Append (P.Buf, Piece);
      for Iter in 1 .. 32 loop
         exit when not Step (P, Sink);
         pragma Unreferenced (Iter);
      end loop;
   end Feed;

   procedure Finalize (P : in out Parser;
                       Sink : access LLM_Qwen.Chat_Sink'Class) is
   begin
      declare
         Saved : constant State_Kinds := P.State;
         Str   : constant String := To_String (P.Buf);
      begin
         if Saved = S_In_Tool then
            --  Tool block still open at end-of-stream. The Hura fine-tune
            --  routinely ends the turn right after </function> WITHOUT the
            --  outer </tool_call>, so Step never fired the call. Recover it:
            --  if the buffer holds a <function=NAME>...</function>, emit it as
            --  a tool call (streaming clients then see the tool_call event and
            --  the run proceeds instead of failing with "empty response"). Only
            --  when there is no function block do we fall back to treating the
            --  bytes as answer text. DO NOT mark Text_After_Last_Tool.
            if not Emit_Tool_Block (P, Str, Sink) then
               P.Answer := P.Answer & Str;
            end if;
            P.Buf := Null_Unbounded_String;
         elsif Str'Length > 0 then
            Emit_Text (P, Str, Sink);
         end if;
      end;

      declare
         R : constant String :=
           (if P.N_Calls > 0 and not P.Text_After_Last_Tool
            then "tool_calls" else "stop");
      begin
         P.Finish := To_Unbounded_String (R);
         LLM_Qwen.On_Finish_Reason (Sink.all, R);
      end;
   end Finalize;

   function Reasoning_Of (P : Parser) return String is
     (To_String (P.Reasoning));

   function Answer_Of (P : Parser) return String is
     (To_String (P.Answer));

   function Finish_Of (P : Parser) return String is
     (To_String (P.Finish));

   function Tool_Calls_Of (P : Parser) return Tool_Call_Array is
      subtype R is Tool_Call_Array (1 .. P.N_Calls);
   begin
      if P.N_Calls = 0 then
         return R'(1 .. 0 => <>);
      end if;
      return R (P.Calls (1 .. P.N_Calls));
   end Tool_Calls_Of;

   function N_Tool_Calls (P : Parser) return Natural is
     (P.N_Calls);

end LLM_Chat_Parser;
