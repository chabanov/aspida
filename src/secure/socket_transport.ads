---------------------------------------------------------------------
-- Socket_Transport — Secure_Channel.Byte_Transport over a TCP socket
-- (GNAT.Sockets). Write sends all bytes; Read blocks until the buffer is
-- filled, raising Connection_Closed if the peer hangs up.
---------------------------------------------------------------------

with Crypto;
with Secure_Channel;
with GNAT.Sockets;

package Socket_Transport is

   Connection_Closed : exception;

   type Sock_Transport is limited new Secure_Channel.Byte_Transport with record
      Sock : GNAT.Sockets.Socket_Type;
   end record;

   overriding procedure Write
     (T : in out Sock_Transport; Data : Crypto.Byte_Array);
   overriding procedure Read
     (T : in out Sock_Transport; Data : out Crypto.Byte_Array);

end Socket_Transport;
