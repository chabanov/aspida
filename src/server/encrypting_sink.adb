---------------------------------------------------------------------
-- Encrypting_Sink body
---------------------------------------------------------------------

with Crypto;     use Crypto;
with Protocol;

package body Encrypting_Sink is

   overriding procedure Emit (S : in out Enc_Sink; Piece : String) is
      Msg : Byte_Array (0 .. Piece'Length);
   begin
      Msg (0) := Protocol.Tag_Token;
      for I in Piece'Range loop
         Msg (I - Piece'First + 1) := U8 (Character'Pos (Piece (I)));
      end loop;
      Secure_Channel.Send_Message (S.Ch.all, S.T, Msg);
   end Emit;

   overriding procedure Tick (S : in out Enc_Sink) is
      Msg : constant Byte_Array := [0 => Protocol.Tag_Prefill];
   begin
      Secure_Channel.Send_Message (S.Ch.all, S.T, Msg);
   end Tick;

end Encrypting_Sink;
