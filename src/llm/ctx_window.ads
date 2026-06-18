---------------------------------------------------------------------
-- Ctx_Window — turn-aware context-window fitting (pure, backend-agnostic).
--
-- Decides which conversation messages to keep when the full transcript would
-- overflow the model's context window. Strategy (StreamingLLM-style + a
-- sliding window of turns, as used by llama.cpp / production chat clients):
--   * ALWAYS keep a leading system prompt (message 1, if System_First) — it
--     carries the model's instructions and acts as an attention sink;
--   * ALWAYS keep the newest message (the current user turn);
--   * then fill the remaining budget with the most RECENT turns, dropping the
--     oldest middle turns first; original order is preserved.
-- Overflow is reported (rather than silently mangling) when even the system
-- prompt + the newest turn + fixed overhead do not fit.
---------------------------------------------------------------------

package Ctx_Window is

   type Len_Array  is array (Positive range <>) of Natural;
   type Keep_Array is array (Positive range <>) of Boolean;

   --  Lengths (I) = token count of message I, in conversation order (1 = oldest).
   --  System_First = message 1 is a system prompt to pin. Overhead = fixed
   --  tokens that are always present (e.g. BOS + the trailing assistant header).
   --  Budget = max prompt tokens (window minus the reserved generation room).
   --  Keep'Range must equal Lengths'Range. Overflow => essentials don't fit.
   procedure Select_Messages
     (Lengths      : Len_Array;
      System_First : Boolean;
      Overhead     : Natural;
      Budget       : Natural;
      Keep         : out Keep_Array;
      Overflow     : out Boolean)
     with Pre => Keep'First = Lengths'First and then Keep'Last = Lengths'Last;

end Ctx_Window;
