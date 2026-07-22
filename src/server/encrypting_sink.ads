---------------------------------------------------------------------
-- Encrypting_Sink — an LLM_Qwen.Chat_Sink that encrypts every chat event
-- (text piece, prefill tick, reasoning piece, assembled tool call, finish
-- reason) as a Secure_Channel record and streams it to the client in real
-- time. Plugs straight into LLM_Engine.Chat's Sink parameter, so the
-- engine itself never touches the network or ciphertext.
--
-- Wire shape (one-byte tag, then UTF-8 body):
--   Tag_Prefill         — emit once per prompt token during prefill
--   Tag_Reasoning_Begin — first piece of a ① block; opens reasoning_content
--                          on the proxy
--   Tag_Text_Begin      — first piece of the assistant's final text (after
--                          any ①/tool_call sequence); opens content on proxy
--   Tag_Token           — one text piece (inside reasoning OR assistant text)
--   Tag_Tool_Call       — body = JSON {"id","name","arguments"}
--   Tag_Finish_Reason   — body = "stop" | "length" | "tool_calls"
--   Tag_Done            — terminal record (carries usage if Stats /= null)
---------------------------------------------------------------------
with LLM_Qwen;
with Secure_Channel;

package Encrypting_Sink is

   type Enc_Sink is new LLM_Qwen.Chat_Sink with record
      Ch      : access Secure_Channel.Channel;
      T       : access Secure_Channel.Byte_Transport'Class;
      --  Last channel emitted: 0 = Text, 1 = Reasoning, 2 = Tool (we only use
      --  Text vs Reasoning to suppress redundant open markers).
      In_Reasoning : Boolean := False;
      --  Set once a send fails (client hung up mid-stream). Further sends are
      --  skipped and the generation loop stops cleanly via Stop_Requested
      --  instead of the sink raising and corrupting the batch lane.
      Client_Gone  : Boolean := False;
   end record;

   --  True after the client disconnected: the engine loop exits cleanly.
   overriding function Stop_Requested (S : Enc_Sink) return Boolean;

   overriding procedure Emit           (S : in out Enc_Sink; Piece : String);
   overriding procedure Tick           (S : in out Enc_Sink);
   overriding procedure On_Reasoning   (S : in out Enc_Sink; Piece : String);
   overriding procedure On_Text        (S : in out Enc_Sink; Piece : String);
   overriding procedure On_Tool_Call
     (S : in out Enc_Sink;
      Id            : String;
      Name          : String;
      Arguments_JS  : String);
   overriding procedure On_Finish_Reason
     (S : in out Enc_Sink; Reason : String);

end Encrypting_Sink;
