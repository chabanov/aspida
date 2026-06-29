---------------------------------------------------------------------
-- Aspida_CAPI — C ABI over the LLM engine (libaspida.dylib).
--
-- Exposes LLM_Engine.Load / Unload / Chat / Detect_Arch / Supports and
-- LLM_Catalog.Discover as C-callable functions with `pragma Export
-- (Convention => C)`. The Swift (or any C) side includes aspida.h and
-- links libaspida.dylib.
--
-- Streaming is bridged via a plain C struct of function pointers
-- (aspida_sink_t): the body builds a concrete LLM_Qwen.Chat_Sink that
-- forwards each event to the matching C callback. No Ada exception
-- ever crosses the C boundary — every wrapper has an `exception when
-- others` that records the message in the thread-local error buffer
-- and returns a failure value (NULL / 0).
--
-- Memory ownership: functions returning chars_ptr own a NUL-terminated
-- string the caller must release with Aspida_Free_String. The opaque
-- engine handle is a System.Address (an access all LLM_Engine.Engine
-- allocated on the heap in the body).
---------------------------------------------------------------------

with Interfaces.C;
with Interfaces.C.Strings;
with System;

package Aspida_CAPI is

   ------------------------------------------------------------------
   --  C-facing types (all Convention C). Mirrored in include/aspida.h.
   ------------------------------------------------------------------

   --  Role (aspida_role_t)
   type Role_Kind_C is (Role_System_C, Role_User_C, Role_Assistant_C);
   for Role_Kind_C use
     (Role_System_C => 0, Role_User_C => 1, Role_Assistant_C => 2);
   for Role_Kind_C'Size use 32;

   --  Message (aspida_message_t): { role, text* }
   type Message_C is record
      Role : Role_Kind_C;
      Text : Interfaces.C.Strings.chars_ptr;  --  const char*, NUL-terminated
   end record;
   pragma Convention (C, Message_C);

   --  Sampling params (aspida_params_t), passed by value.
   type Params_C is record
      Temperature    : Interfaces.C.C_Float := 0.0;
      Top_K          : Interfaces.C.int     := 0;
      Top_P          : Interfaces.C.C_Float := 1.0;
      Min_P          : Interfaces.C.C_Float := 0.0;
      Repeat_Penalty : Interfaces.C.C_Float := 1.0;
      Repeat_Last_N  : Interfaces.C.int     := 64;
      Min_Tokens     : Interfaces.C.int     := 0;
      Seed           : Interfaces.C.long    := 0;
   end record;
   pragma Convention (C, Params_C);

   --  Per-generation accounting out (aspida_stats_t). Booleans are 0/1 ints.
   type Stats_C is record
      Prompt_Tokens     : Interfaces.C.int := 0;
      Completion_Tokens : Interfaces.C.int := 0;
      Truncated         : Interfaces.C.int := 0;
      Overflow          : Interfaces.C.int := 0;
   end record;
   pragma Convention (C, Stats_C);

   --  Streaming callbacks. Each receives the opaque user_data pointer that
   --  was in aspida_sink_t.user_data. Piece strings are NUL-terminated and
   --  valid only for the duration of the callback (do not retain).
   subtype User_Data_T is System.Address;
   Null_User_Data : constant User_Data_T := System.Null_Address;

   type On_Tick_Callback is access procedure (UD : User_Data_T);
   pragma Convention (C, On_Tick_Callback);

   type On_Piece_Callback is access procedure
     (Piece : Interfaces.C.Strings.chars_ptr; UD : User_Data_T);
   pragma Convention (C, On_Piece_Callback);

   type On_Tool_Call_Callback is access procedure
     (Id   : Interfaces.C.Strings.chars_ptr;
      Name : Interfaces.C.Strings.chars_ptr;
      Args : Interfaces.C.Strings.chars_ptr;
      UD   : User_Data_T);
   pragma Convention (C, On_Tool_Call_Callback);

   type On_Finish_Callback is access procedure
     (Reason : Interfaces.C.Strings.chars_ptr; UD : User_Data_T);
   pragma Convention (C, On_Finish_Callback);

   --  aspida_sink_t. Any callback may be null (System.Null_Address).
   type Sink_C is record
      On_Tick      : On_Tick_Callback      := null;
      On_Reasoning : On_Piece_Callback     := null;
      On_Text      : On_Piece_Callback     := null;
      On_Tool_Call : On_Tool_Call_Callback := null;
      On_Finish    : On_Finish_Callback    := null;
      User_Data    : User_Data_T           := Null_User_Data;
   end record;
   pragma Convention (C, Sink_C);

   ------------------------------------------------------------------
   --  Exported functions
   ------------------------------------------------------------------

   --  One-time/per-thread initialization. The Ada runtime must attach the
   --  calling (foreign, non-Ada) thread before any API call that uses the
   --  secondary stack (every function below except Unload/Last_Error/
   --  Free_String). Aspida_Init does this for the calling thread; it is also
   --  called automatically inside each wrapper, so hosts may ignore it. Safe
   --  to call repeatedly. Returns 1 on success, 0 on failure.
   function Aspida_Init return Interfaces.C.int;
   pragma Export (Convention => C, Entity => Aspida_Init,
                  External_Name => "aspida_init");

   --  Load a model from a GGUF path. Returns an opaque engine handle, or
   --  Null_Address on failure (see Aspida_Last_Error). The handle owns the
   --  backend; release it with Aspida_Unload.
   function Aspida_Load (Path : Interfaces.C.Strings.chars_ptr)
                         return System.Address;
   pragma Export (Convention => C, Entity => Aspida_Load,
                  External_Name => "aspida_load");

   --  Unload and free an engine. Null-safe and idempotent.
   procedure Aspida_Unload (E : System.Address);
   pragma Export (Convention => C, Entity => Aspida_Unload,
                  External_Name => "aspida_unload");

   --  Last error message for the calling thread. The returned pointer is
   --  valid until the next API call sets a new error; do NOT free it.
   function Aspida_Last_Error return Interfaces.C.Strings.chars_ptr;
   pragma Export (Convention => C, Entity => Aspida_Last_Error,
                  External_Name => "aspida_last_error");

   --  Peek a GGUF's general.architecture without loading weights.
   --  Returns a freshly allocated string ("" if unreadable); caller frees
   --  with Aspida_Free_String.
   function Aspida_Detect_Arch (Path : Interfaces.C.Strings.chars_ptr)
                                return Interfaces.C.Strings.chars_ptr;
   pragma Export (Convention => C, Entity => Aspida_Detect_Arch,
                  External_Name => "aspida_detect_arch");

   --  1 if the GGUF's architecture is supported by the engine, else 0.
   function Aspida_Arch_Supported (Path : Interfaces.C.Strings.chars_ptr)
                                   return Interfaces.C.int;
   pragma Export (Convention => C, Entity => Aspida_Arch_Supported,
                  External_Name => "aspida_arch_supported");

   --  Enumerate GGUF models. Dirs is a ':'-joined list of search roots, or
   --  Null_Ptr to use the default roots (ASPIDA_MODELS_DIR + common dirs).
   --  Returns a JSON array string; caller frees with Aspida_Free_String:
   --    [{"path","name","arch","quant","params","size","supported":bool}]
   function Aspida_Discover_Models (Dirs : Interfaces.C.Strings.chars_ptr)
                                    return Interfaces.C.Strings.chars_ptr;
   pragma Export (Convention => C, Entity => Aspida_Discover_Models,
                  External_Name => "aspida_discover_models");

   --  Run a chat turn. Messages points to N consecutive Message_C structs.
   --  Sink may be null (System.Null_Address) for non-streaming. Stats may be
   --  null to skip accounting. Returns a JSON result string on success:
   --    {"reasoning","answer","finish","tool_calls":[{"id","name","arguments"}],
   --     "usage":{"prompt_tokens","completion_tokens","truncated","overflow"}}
   --  Returns Null_Ptr on failure (see Aspida_Last_Error). Caller frees the
   --  success result with Aspida_Free_String.
   function Aspida_Chat
     (E              : System.Address;
      Messages       : access constant Message_C;
      N              : Interfaces.C.int;
      Max_New_Tokens : Interfaces.C.int;
      Params         : access constant Params_C;
      Sink           : access constant Sink_C;
      Stats          : access Stats_C)
      return Interfaces.C.Strings.chars_ptr;
   pragma Export (Convention => C, Entity => Aspida_Chat,
                  External_Name => "aspida_chat");

   --  Engine metadata.
   function Aspida_Vocab_Size (E : System.Address) return Interfaces.C.int;
   pragma Export (Convention => C, Entity => Aspida_Vocab_Size,
                  External_Name => "aspida_vocab_size");

   function Aspida_Arch_Name (E : System.Address)
                              return Interfaces.C.Strings.chars_ptr;
   pragma Export (Convention => C, Entity => Aspida_Arch_Name,
                  External_Name => "aspida_arch_name");

   --  Release a string returned by Aspida_Detect_Arch / Discover_Models /
   --  Chat / Arch_Name. No-op on Null_Ptr.
   procedure Aspida_Free_String (S : Interfaces.C.Strings.chars_ptr);
   pragma Export (Convention => C, Entity => Aspida_Free_String,
                  External_Name => "aspida_free_string");

end Aspida_CAPI;