---------------------------------------------------------------------
-- Aspida_CAPI body — see spec for the contract.
--
-- Implementation notes
--   * The opaque engine handle is a heap `access all LLM_Engine.Engine`,
--     converted to/from System.Address via System.Address_To_Access_
--     Conversions (the standard C-interop generic; LLM_Engine.Engine is
--     private but usable as the generic's `type Object is private`).
--   * C_Sink is a concrete LLM_Qwen.Chat_Sink that copies the C sink struct
--     and dispatches every event to the matching C function pointer,
--     allocating a transient chars_ptr per piece and freeing it after the
--     callback returns (the C side must not retain it).
--   * Error reporting uses a package-level buffer (v1: not thread-safe —
--     the host app serializes API calls per engine). Every wrapper catches
--     all exceptions and records the message; an Ada exception never crosses
--     the C boundary.
---------------------------------------------------------------------

with Ada.Exceptions;
with Ada.Environment_Variables;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with Ada.Task_Identification;
with Ada.Unchecked_Deallocation;
with LLM_Catalog;
with LLM_Engine;
with LLM_Qwen;
with LLM_Sampler;
with System.Address_To_Access_Conversions;

package body Aspida_CAPI is

   use type Interfaces.C.int;
   use type Interfaces.C.Strings.chars_ptr;
   use type System.Address;

   package SU renames Ada.Strings.Unbounded;
   package CS renames Interfaces.C.Strings;

   ------------------------------------------------------------------
   --  Engine handle <-> access conversion
   ------------------------------------------------------------------
   package Engine_Ptr is new System.Address_To_Access_Conversions
     (Object => LLM_Engine.Engine);
   use Engine_Ptr;
   procedure Free_Engine is new Ada.Unchecked_Deallocation
     (Object => LLM_Engine.Engine, Name => Engine_Ptr.Object_Pointer);

   ------------------------------------------------------------------
   --  Error buffer (v1: single-threaded; see header note)
   ------------------------------------------------------------------
   Error_Buf : CS.chars_ptr := CS.Null_Ptr;

   procedure Set_Error (Msg : String) is
      Tmp : CS.chars_ptr := Error_Buf;
   begin
      Error_Buf := CS.Null_Ptr;
      if Tmp /= CS.Null_Ptr then
         CS.Free (Tmp);
      end if;
      Error_Buf := CS.New_String (Msg);
   end Set_Error;

   ------------------------------------------------------------------
   --  Foreign-thread attachment.
   --
   --  The Ada secondary stack (used by every function returning an
   --  unconstrained String) is allocated per Ada-task in the thread's
   --  Thread-Specific Data. A foreign (non-Ada) thread that calls into
   --  the library has no TSD until it is registered as an Ada task. GNAT
   --  auto-registers via System.Task_Primitives.Operations.Self, which
   --  runs Create_TSD (allocating the sec stack) — but the sec-stack
   --  soft-link Get_Sec_Stack reads the TSD *directly* and never calls
   --  Self, so an unregistered thread gets a null stack and crashes in
   --  SS_Mark. Touching Ada.Task_Identification.Current_Task forces Self,
   --  which registers the calling thread if needed (cheap & idempotent:
   --  Self checks pthread_getspecific first). Call this at the top of
   --  every wrapper that may run engine code.
   ------------------------------------------------------------------
   procedure Ensure_Registered is
      Discard : Ada.Task_Identification.Task_Id;
      pragma Unreferenced (Discard);
   begin
      Discard := Ada.Task_Identification.Current_Task;
   exception
      when others =>
         null;  --  best effort; the call still leaves the thread registered
   end Ensure_Registered;

   ------------------------------------------------------------------
   --  chars_ptr helpers
   ------------------------------------------------------------------
   function To_Ada (C : CS.chars_ptr) return String is
   begin
      if C = CS.Null_Ptr then
         return "";
      end if;
      return CS.Value (C);
   end To_Ada;

   function To_New_C (S : String) return CS.chars_ptr is
     (CS.New_String (S));

   ------------------------------------------------------------------
   --  Minimal JSON string escaping (RFC 8259).
   ------------------------------------------------------------------
   function Hex (N : Natural) return Character is
      (Character'Val (if N < 10 then N + Character'Pos ('0')
                                  else N - 10 + Character'Pos ('a')));
   function Esc (S : String) return String is
      R : SU.Unbounded_String;
   begin
      for Ch of S loop
         case Ch is
            when '"'  => SU.Append (R, "\""");   --  JSON \" (backslash + quote)
            when '\'  => SU.Append (R, "\\");
            when ASCII.LF => SU.Append (R, "\n");
            when ASCII.CR => SU.Append (R, "\r");
            when ASCII.HT => SU.Append (R, "\t");
            when ASCII.BS => SU.Append (R, "\b");
            when ASCII.FF => SU.Append (R, "\f");
            when others =>
               if Character'Pos (Ch) < 32 then
                  SU.Append (R, "\u00");
                  SU.Append (R, Hex (Character'Pos (Ch) / 16));
                  SU.Append (R, Hex (Character'Pos (Ch) mod 16));
               else
                  SU.Append (R, Ch);
               end if;
         end case;
      end loop;
      return SU.To_String (R);
   end Esc;

   ------------------------------------------------------------------
   --  C_Sink: a Chat_Sink that forwards to the C function pointers.
   ------------------------------------------------------------------
   type C_Sink is new LLM_Qwen.Chat_Sink with record
      C : Sink_C;
   end record;

   overriding procedure Tick           (S : in out C_Sink);
   overriding procedure Emit           (S : in out C_Sink; Piece : String);
   overriding procedure On_Reasoning    (S : in out C_Sink; Piece : String);
   overriding procedure On_Text          (S : in out C_Sink; Piece : String);
   overriding procedure On_Tool_Call
     (S : in out C_Sink; Id, Name, Arguments_JS : String);
   overriding procedure On_Finish_Reason (S : in out C_Sink; Reason : String);

   overriding procedure Tick (S : in out C_Sink) is
   begin
      if S.C.On_Tick /= null then
         S.C.On_Tick (S.C.User_Data);
      end if;
   end Tick;

   --  The engine streams each answer token via Token_Sink.Emit. The base
   --  Chat_Sink.Emit forwards to On_Text, but that forward is statically
   --  bound to the base Chat_Sink.On_Text (a specific-type controlling
   --  parameter does not redispatch), so a sink overriding only On_Text would
   --  never receive text. Override Emit here to call our own On_Text (with
   --  S : in out C_Sink this statically resolves to C_Sink.On_Text).
   overriding procedure Emit (S : in out C_Sink; Piece : String) is
   begin
      On_Text (S, Piece);
   end Emit;

   overriding procedure On_Reasoning (S : in out C_Sink; Piece : String) is
   begin
      if S.C.On_Reasoning /= null then
         declare
            P : CS.chars_ptr := CS.New_String (Piece);
         begin
            S.C.On_Reasoning (P, S.C.User_Data);
            CS.Free (P);
         end;
      end if;
   end On_Reasoning;

   overriding procedure On_Text (S : in out C_Sink; Piece : String) is
   begin
      if S.C.On_Text /= null then
         declare
            P : CS.chars_ptr := CS.New_String (Piece);
         begin
            S.C.On_Text (P, S.C.User_Data);
            CS.Free (P);
         end;
      end if;
   end On_Text;

   overriding procedure On_Tool_Call
     (S : in out C_Sink; Id, Name, Arguments_JS : String)
   is
   begin
      if S.C.On_Tool_Call /= null then
         declare
            CI : CS.chars_ptr := CS.New_String (Id);
            CN : CS.chars_ptr := CS.New_String (Name);
            CA : CS.chars_ptr := CS.New_String (Arguments_JS);
         begin
            S.C.On_Tool_Call (CI, CN, CA, S.C.User_Data);
            CS.Free (CI);
            CS.Free (CN);
            CS.Free (CA);
         end;
      end if;
   end On_Tool_Call;

   overriding procedure On_Finish_Reason (S : in out C_Sink; Reason : String) is
   begin
      if S.C.On_Finish /= null then
         declare
            R : CS.chars_ptr := CS.New_String (Reason);
         begin
            S.C.On_Finish (R, S.C.User_Data);
            CS.Free (R);
         end;
      end if;
   end On_Finish_Reason;

   ------------------------------------------------------------------
   --  Build the JSON result string for a Chat call.
   ------------------------------------------------------------------
   function Bool_Int (B : Boolean) return String is (if B then "true" else "false");

   function Build_Result
     (R : LLM_Qwen.Chat_Result; St : LLM_Qwen.Gen_Stats) return String
   is
      J : SU.Unbounded_String;
   begin
      J := SU.To_Unbounded_String ("{""reasoning"":""");
      SU.Append (J, Esc (SU.To_String (R.Reasoning)));
      SU.Append (J, """,""answer"":""");
      SU.Append (J, Esc (SU.To_String (R.Answer)));
      SU.Append (J, """,""finish"":""");
      SU.Append (J, Esc (SU.To_String (R.Finish)));
      SU.Append (J, """,""tool_calls"":[");

      for I in 1 .. R.N_Tool_Calls loop
         if I > 1 then
            SU.Append (J, ",");
         end if;
         SU.Append (J, "{""id"":""");
         SU.Append (J, Esc (SU.To_String (R.Tool_Calls (I).Id)));
         SU.Append (J, """,""name"":""");
         SU.Append (J, Esc (SU.To_String (R.Tool_Calls (I).Name)));
         SU.Append (J, """,""arguments"":""");
         SU.Append (J, Esc (SU.To_String (R.Tool_Calls (I).Arguments_JS)));
         SU.Append (J, """}");
      end loop;

      SU.Append (J, "],""usage"":{""prompt_tokens"":");
      SU.Append (J, Ada.Strings.Fixed.Trim (St.Prompt_Tokens'Image,
                                           Ada.Strings.Both));
      SU.Append (J, ",""completion_tokens"":");
      SU.Append (J, Ada.Strings.Fixed.Trim (St.Completion_Tokens'Image,
                                           Ada.Strings.Both));
      SU.Append (J, ",""truncated"":");
      SU.Append (J, Bool_Int (St.Truncated));
      SU.Append (J, ",""overflow"":");
      SU.Append (J, Bool_Int (St.Overflow));
      SU.Append (J, "}}");
      return SU.To_String (J);
   end Build_Result;

   function Build_Catalog (V : LLM_Catalog.Entry_Vectors.Vector) return String is
      J : SU.Unbounded_String;
      First : Boolean := True;

      function Status_Str (S : LLM_Catalog.Model_Status) return String is
        (case S is
            when LLM_Catalog.Supported   => "supported",
            when LLM_Catalog.Unsupported => "unsupported",
            when LLM_Catalog.Projector   => "projector",
            when LLM_Catalog.Invalid      => "invalid");
   begin
      J := SU.To_Unbounded_String ("[");
      for E of V loop
         if not First then
            SU.Append (J, ",");
         end if;
         First := False;
         SU.Append (J, "{""path"":""");
         SU.Append (J, Esc (SU.To_String (E.Path)));
         SU.Append (J, """,""name"":""");
         SU.Append (J, Esc (SU.To_String (E.Name)));
         SU.Append (J, """,""arch"":""");
         SU.Append (J, Esc (SU.To_String (E.Arch)));
         SU.Append (J, """,""quant"":""");
         SU.Append (J, Esc (SU.To_String (E.Quant)));
         SU.Append (J, """,""params"":""");
         SU.Append (J, Esc (SU.To_String (E.Params)));
         SU.Append (J, """,""size"":");
         SU.Append (J, Ada.Strings.Fixed.Trim (E.Size'Image, Ada.Strings.Both));
         SU.Append (J, ",""supported"":");
         SU.Append (J, Bool_Int (E.Status in LLM_Catalog.Supported));
         SU.Append (J, ",""status"":""");
         SU.Append (J, Status_Str (E.Status));
         SU.Append (J, """}");
      end loop;
      SU.Append (J, "]");
      return SU.To_String (J);
   end Build_Catalog;

   ------------------------------------------------------------------
   --  Exported functions
   ------------------------------------------------------------------

   function Aspida_Init return Interfaces.C.int is
   begin
      Ensure_Registered;
      return 1;
   exception
      when others =>
         return 0;
   end Aspida_Init;

   function Aspida_Load (Path : CS.chars_ptr) return System.Address is
   begin
      Ensure_Registered;
      if Path = CS.Null_Ptr then
         Set_Error ("aspida_load: null path");
         return System.Null_Address;
      end if;
      declare
         E_Acc : Engine_Ptr.Object_Pointer := new LLM_Engine.Engine;
      begin
         E_Acc.all := LLM_Engine.Load (CS.Value (Path));
         return Engine_Ptr.To_Address (E_Acc);
      exception
         when E : others =>
            begin
               Free_Engine (E_Acc);
            exception
               when others => null;
            end;
            Set_Error ("aspida_load: " & Ada.Exceptions.Exception_Message (E));
            return System.Null_Address;
      end;
   end Aspida_Load;

   procedure Aspida_Unload (E : System.Address) is
      E_Acc : Engine_Ptr.Object_Pointer;
   begin
      if E = System.Null_Address then
         return;
      end if;
      E_Acc := Engine_Ptr.To_Pointer (E);
      if E_Acc = null then
         return;
      end if;
      begin
         LLM_Engine.Unload (E_Acc.all);
      exception
         when others =>
            null;  --  best-effort teardown; still free the cell below
      end;
      Free_Engine (E_Acc);
   exception
      when others =>
         null;
   end Aspida_Unload;

   function Aspida_Last_Error return CS.chars_ptr is
     (Error_Buf);

   function Aspida_Detect_Arch (Path : CS.chars_ptr) return CS.chars_ptr is
   begin
      Ensure_Registered;
      if Path = CS.Null_Ptr then
         return To_New_C ("");
      end if;
      declare
         Arch : constant String := LLM_Engine.Detect_Arch (CS.Value (Path));
      begin
         return To_New_C (Arch);
      exception
         when E : others =>
            Set_Error ("aspida_detect_arch: " & Ada.Exceptions.Exception_Message (E));
            return To_New_C ("");
      end;
   end Aspida_Detect_Arch;

   function Aspida_Arch_Supported (Path : CS.chars_ptr) return Interfaces.C.int is
   begin
      Ensure_Registered;
      if Path = CS.Null_Ptr then
         return 0;
      end if;
      declare
         Arch : constant String := LLM_Engine.Detect_Arch (CS.Value (Path));
      begin
         return (if LLM_Engine.Supports (Arch) then 1 else 0);
      exception
         when E : others =>
            Set_Error ("aspida_arch_supported: " & Ada.Exceptions.Exception_Message (E));
            return 0;
      end;
   end Aspida_Arch_Supported;

   function Aspida_Discover_Models (Dirs : CS.chars_ptr) return CS.chars_ptr is
   begin
      Ensure_Registered;
      if Dirs /= CS.Null_Ptr then
         Ada.Environment_Variables.Set ("ASPIDA_MODELS_DIR", CS.Value (Dirs));
      end if;
      declare
         V : constant LLM_Catalog.Entry_Vectors.Vector := LLM_Catalog.Discover;
      begin
         return To_New_C (Build_Catalog (V));
      exception
         when E : others =>
            Set_Error ("aspida_discover_models: " & Ada.Exceptions.Exception_Message (E));
            return To_New_C ("[]");
      end;
   end Aspida_Discover_Models;

   function Aspida_Chat
     (E              : System.Address;
      Messages       : access constant Message_C;
      N              : Interfaces.C.int;
      Max_New_Tokens : Interfaces.C.int;
      Params         : access constant Params_C;
      Sink           : access constant Sink_C;
      Stats          : access Stats_C)
      return CS.chars_ptr
   is
      E_Acc : Engine_Ptr.Object_Pointer;
   begin
      Ensure_Registered;
      if E = System.Null_Address then
         Set_Error ("aspida_chat: null engine");
         return CS.Null_Ptr;
      end if;
      E_Acc := Engine_Ptr.To_Pointer (E);
      if E_Acc = null then
         Set_Error ("aspida_chat: invalid engine handle");
         return CS.Null_Ptr;
      end if;
      if N <= 0 or else Messages = null then
         Set_Error ("aspida_chat: no messages");
         return CS.Null_Ptr;
      end if;
      if Params = null then
         Set_Error ("aspida_chat: null params");
         return CS.Null_Ptr;
      end if;

      declare
         M : constant Integer := Integer (N);

         --  Overlay the C message array (Message_C is Convention C).
         type Arr_T is array (Integer range <>) of aliased Message_C;
         pragma Convention (C, Arr_T);
         CArr : Arr_T (0 .. M - 1);
         for CArr'Address use Messages.all'Address;
         pragma Import (Ada, CArr);

         --  Build the Ada conversation (1-based, matching Message_Array).
         Conv : LLM_Qwen.Message_Array (1 .. M);

         P : constant LLM_Sampler.Params :=
           (Temperature    => Float (Params.all.Temperature),
            Top_K          => Integer (Params.all.Top_K),
            Top_P          => Float (Params.all.Top_P),
            Min_P          => Float (Params.all.Min_P),
            Repeat_Penalty => Float (Params.all.Repeat_Penalty),
            Repeat_Last_N  => Integer (Params.all.Repeat_Last_N),
            Min_Tokens     => Integer (Params.all.Min_Tokens),
            Seed           => Long_Long_Integer (Params.all.Seed));

         Stats_Obj : aliased LLM_Qwen.Gen_Stats;
         Stats_Ptr : constant access LLM_Qwen.Gen_Stats :=
           (if Stats = null then null else Stats_Obj'Access);

         --  Chat_Result has a non-default discriminant (N_Tool_Calls), so it
         --  must be initialised at declaration. Wrap the engine call (which
         --  may dispatch through the streaming sink) in this local function,
         --  invoked AFTER Conv is filled.
         function Call return LLM_Qwen.Chat_Result is
         begin
            if Sink = null then
               return LLM_Engine.Chat
                 (E_Acc.all, Conv, Integer (Max_New_Tokens),
                  null, P, Stats_Ptr);
            else
               declare
                  Sink_Obj : aliased C_Sink;
               begin
                  Sink_Obj.C := Sink.all;
                  return LLM_Engine.Chat
                    (E_Acc.all, Conv, Integer (Max_New_Tokens),
                     Sink_Obj'Access, P, Stats_Ptr);
               end;
            end if;
         end Call;
      begin
         for I in 0 .. M - 1 loop
            case CArr (I).Role is
               when Role_System_C     => Conv (I + 1).Role := LLM_Qwen.Role_System;
               when Role_User_C       => Conv (I + 1).Role := LLM_Qwen.Role_User;
               when Role_Assistant_C  => Conv (I + 1).Role := LLM_Qwen.Role_Assistant;
            end case;
            Conv (I + 1).Text := SU.To_Unbounded_String (To_Ada (CArr (I).Text));
         end loop;

         declare
            Result : constant LLM_Qwen.Chat_Result := Call;
         begin
            if Stats /= null then
               Stats.all := (Prompt_Tokens     => Interfaces.C.int (Stats_Obj.Prompt_Tokens),
                             Completion_Tokens => Interfaces.C.int (Stats_Obj.Completion_Tokens),
                             Truncated         => (if Stats_Obj.Truncated then 1 else 0),
                             Overflow          => (if Stats_Obj.Overflow then 1 else 0));
            end if;
            return To_New_C (Build_Result (Result, Stats_Obj));
         end;
      exception
         when E : others =>
            Set_Error ("aspida_chat: " & Ada.Exceptions.Exception_Message (E));
            return CS.Null_Ptr;
      end;
   end Aspida_Chat;

   function Aspida_Vocab_Size (E : System.Address) return Interfaces.C.int is
   begin
      Ensure_Registered;
      if E = System.Null_Address then
         return 0;
      end if;
      declare
         E_Acc : constant Engine_Ptr.Object_Pointer := Engine_Ptr.To_Pointer (E);
      begin
         if E_Acc = null then
            return 0;
         end if;
         return Interfaces.C.int (LLM_Engine.Vocab_Size (E_Acc.all));
      exception
         when others =>
            return 0;
      end;
   end Aspida_Vocab_Size;

   function Aspida_Arch_Name (E : System.Address) return CS.chars_ptr is
   begin
      Ensure_Registered;
      if E = System.Null_Address then
         return To_New_C ("");
      end if;
      declare
         E_Acc : constant Engine_Ptr.Object_Pointer := Engine_Ptr.To_Pointer (E);
      begin
         if E_Acc = null then
            return To_New_C ("");
         end if;
         return To_New_C (LLM_Engine.Arch_Name (E_Acc.all));
      exception
         when E : others =>
            Set_Error ("aspida_arch_name: " & Ada.Exceptions.Exception_Message (E));
            return To_New_C ("");
      end;
   end Aspida_Arch_Name;

   procedure Aspida_Free_String (S : CS.chars_ptr) is
      T : CS.chars_ptr := S;
   begin
      if T /= CS.Null_Ptr then
         CS.Free (T);
      end if;
   end Aspida_Free_String;

end Aspida_CAPI;