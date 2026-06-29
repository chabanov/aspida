---------------------------------------------------------------------
-- Encrypting_Sink body
---------------------------------------------------------------------

with Crypto;          use Crypto;
with Protocol;
with Ada.Strings;     use Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body Encrypting_Sink is

   ----------------------------------------------------------------------
   -- JSON encoding helpers (small, no third-party). Built on top of the
   -- server's JSON package via To_String + raw concatenation; we re-use the
   -- server's JSON module rather than pull a private one in.
   ----------------------------------------------------------------------

   function Escaped (S : String) return String is
      Acc : Unbounded_String := Null_Unbounded_String;
   begin
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
      return To_String (Acc);
   end Escaped;

   function Build_Tool_Call_JSON
     (Id, Name, Arguments_JS : String) return String
   is
   begin
      return
        (ASCII.LF & "{" &
         ASCII.LF & "  ""id"": """      & Escaped (Id)           & """," &
         ASCII.LF & "  ""name"": """    & Escaped (Name)         & """," &
         ASCII.LF & "  ""arguments"": " & Escaped (Arguments_JS) &
         ASCII.LF & "}");
   end Build_Tool_Call_JSON;

   procedure Send_Body (S : Enc_Sink; Tag : U8; Payload : String) is
      Msg : Byte_Array (0 .. Payload'Length);
   begin
      Msg (0) := Tag;
      for I in Payload'Range loop
         Msg (I - Payload'First + 1) := U8 (Character'Pos (Payload (I)));
      end loop;
      Secure_Channel.Send_Message (S.Ch.all, S.T, Msg);
   end Send_Body;

   overriding procedure Emit (S : in out Enc_Sink; Piece : String) is
   begin
      --  Default forwarding from Chat_Sink's base Emit; kept so callers
      --  that still use the legacy streaming API (Emit-only text) keep
      --  working — every Emit is treated as a text piece.
      Encrypting_Sink.On_Text (S, Piece);
   end Emit;

   overriding procedure Tick (S : in out Enc_Sink) is
      Msg : constant Byte_Array (0 .. 0) := [0 => Protocol.Tag_Prefill];
   begin
      Secure_Channel.Send_Message (S.Ch.all, S.T, Msg);
   end Tick;

   overriding procedure On_Reasoning (S : in out Enc_Sink; Piece : String) is
   begin
      --  Emit Tag_Reasoning_Begin only when the channel actually flips
      --  (Text → Reasoning). Inside a single reasoning block we just send
      --  Tag_Token; the proxy's `In_Reasoning` flag stays True until the
      --  first Text piece arrives.
      if not S.In_Reasoning then
         Send_Body (S, Protocol.Tag_Reasoning_Begin, "");
         S.In_Reasoning := True;
      end if;
      Send_Body (S, Protocol.Tag_Token, Piece);
   end On_Reasoning;

   overriding procedure On_Text (S : in out Enc_Sink; Piece : String) is
   begin
      if S.In_Reasoning then
         Send_Body (S, Protocol.Tag_Text_Begin, "");
         S.In_Reasoning := False;
      end if;
      Send_Body (S, Protocol.Tag_Token, Piece);
   end On_Text;

   overriding procedure On_Tool_Call
     (S : in out Enc_Sink;
      Id            : String;
      Name          : String;
      Arguments_JS  : String)
   is
      Payload : constant String := Build_Tool_Call_JSON (Id, Name, Arguments_JS);
   begin
      Send_Body (S, Protocol.Tag_Tool_Call, Payload);
   end On_Tool_Call;

   overriding procedure On_Finish_Reason
     (S : in out Enc_Sink; Reason : String)
   is
   begin
      Send_Body (S, Protocol.Tag_Finish_Reason, Reason);
   end On_Finish_Reason;

end Encrypting_Sink;
