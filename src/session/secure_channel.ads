---------------------------------------------------------------------
-- Secure_Channel — authenticated, forward-secret session over any byte
-- transport (a "Noise-NK"-style handshake + framed AEAD records).
--
-- Handshake (responder = server, known static key; initiator = client):
--   C -> S : e   (client ephemeral public)
--   S -> C : f   (server ephemeral public)
--   shared = es = DH(e, s)   (authenticates the server: needs its static
--                             secret; the client pins its static public)
--            ee = DH(e, f)   (forward secrecy: pure-ephemeral)
--   transcript = SHA256(prologue | s_pub | e | f)
--   PRK  = HKDF-Extract(transcript, es | ee)
--   keys = HKDF-Expand(PRK, "keys", 64) -> K_c2s | K_s2c
--   S -> C : AEAD tag over transcript with K_s2c (server key-confirmation;
--            a MITM without the static secret cannot produce it).
--
-- Records: 4-byte big-endian length-prefixed ChaCha20-Poly1305 frames,
-- each direction keyed separately with a monotonic 64-bit nonce counter
-- (never reused). Recv raises Auth_Error on any tamper.
---------------------------------------------------------------------

with Crypto;
with Crypto.X25519;

package Secure_Channel is

   --  Abstract bidirectional byte transport. Write sends all of Data; Read
   --  fills Data exactly (blocking) or raises. Implemented by a socket
   --  (server) or an in-memory loopback (tests).
   type Byte_Transport is limited interface;
   procedure Write (T : in out Byte_Transport; Data : Crypto.Byte_Array)
      is abstract;
   procedure Read (T : in out Byte_Transport; Data : out Crypto.Byte_Array)
      is abstract;

   type Channel is limited private;

   Handshake_Error : exception;   -- malformed handshake / oversize frame
   Auth_Error      : exception;   -- AEAD tag verification failed

   --  Responder side: authenticates itself with its long-term X25519 secret.
   procedure Server_Handshake
     (Ch     : out Channel;
      T      : access Byte_Transport'Class;
      Static_Secret : Crypto.X25519.Key_256);

   --  Initiator side: pins the server's long-term X25519 public key.
   procedure Client_Handshake
     (Ch     : out Channel;
      T      : access Byte_Transport'Class;
      Server_Public : Crypto.X25519.Key_256);

   procedure Send_Message
     (Ch : in out Channel; T : access Byte_Transport'Class;
      Plaintext : Crypto.Byte_Array);
   function  Recv_Message
     (Ch : in out Channel; T : access Byte_Transport'Class)
      return Crypto.Byte_Array;

   --  Wipe the session keys when the session ends (defence in depth).
   procedure Close (Ch : in out Channel);

private

   subtype Key32 is Crypto.Byte_Array (0 .. 31);

   --  The transport is passed per call, not stored, to avoid anonymous-access
   --  accessibility constraints; the Channel holds only the session keys and
   --  per-direction nonce counters.
   type Channel is limited record
      K_Send : Key32;
      K_Recv : Key32;
      N_Send : Crypto.U64 := 0;
      N_Recv : Crypto.U64 := 0;
      Ready  : Boolean := False;
   end record;

end Secure_Channel;
