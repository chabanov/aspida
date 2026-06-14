---------------------------------------------------------------------
-- Encrypting_Sink — an LLM_Qwen.Token_Sink that encrypts each generated
-- token (and prefill tick) as a Secure_Channel record and streams it to
-- the client in real time. Plugs straight into LLM_Qwen.Chat's Sink
-- parameter, so the engine itself never touches the network or ciphertext.
---------------------------------------------------------------------

with LLM_Qwen;
with Secure_Channel;

package Encrypting_Sink is

   type Enc_Sink is new LLM_Qwen.Token_Sink with record
      Ch : access Secure_Channel.Channel;
      T  : access Secure_Channel.Byte_Transport'Class;
   end record;
   overriding procedure Emit (S : in out Enc_Sink; Piece : String);
   overriding procedure Tick (S : in out Enc_Sink);

end Encrypting_Sink;
