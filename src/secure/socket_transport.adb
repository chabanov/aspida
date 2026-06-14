---------------------------------------------------------------------
-- Socket_Transport body
---------------------------------------------------------------------

with Ada.Streams; use Ada.Streams;

package body Socket_Transport is

   use Crypto;
   use GNAT.Sockets;

   overriding procedure Write
     (T : in out Sock_Transport; Data : Crypto.Byte_Array)
   is
      Buf  : Stream_Element_Array (1 .. Stream_Element_Offset (Data'Length));
      Last : Stream_Element_Offset;
      Sent : Stream_Element_Offset := 0;
   begin
      for I in Data'Range loop
         Buf (Stream_Element_Offset (I - Data'First + 1)) :=
           Stream_Element (Data (I));
      end loop;
      while Sent < Buf'Last loop
         Send_Socket (T.Sock, Buf (Sent + 1 .. Buf'Last), Last);
         if Last < Sent + 1 then
            raise Connection_Closed with "send failed";
         end if;
         Sent := Last;
      end loop;
   end Write;

   overriding procedure Read
     (T : in out Sock_Transport; Data : out Crypto.Byte_Array)
   is
      Buf  : Stream_Element_Array (1 .. Stream_Element_Offset (Data'Length));
      Got  : Stream_Element_Offset := 0;
      Last : Stream_Element_Offset;
   begin
      while Got < Buf'Last loop
         Receive_Socket (T.Sock, Buf (Got + 1 .. Buf'Last), Last);
         if Last < Got + 1 then              -- 0 bytes => peer closed
            raise Connection_Closed with "connection closed";
         end if;
         Got := Last;
      end loop;
      for I in Data'Range loop
         Data (I) :=
           U8 (Buf (Stream_Element_Offset (I - Data'First + 1)));
      end loop;
   end Read;

end Socket_Transport;
