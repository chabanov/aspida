---------------------------------------------------------------------
-- Secure_Channel body
---------------------------------------------------------------------

with Interfaces;
with Crypto.SHA256;
with Crypto.HKDF;
with Crypto.AEAD;
with Crypto.Random;

package body Secure_Channel is

   use Interfaces;
   use Crypto;   -- Crypto is visible via the spec's with clause

   Prologue : constant String :=
     "aspida-secure-channel/1 X25519-ChaCha20Poly1305-HKDF-SHA256";
   Info     : constant String := "keys";

   Max_Frame : constant := 1_048_576;   -- 1 MiB cap (anti-DoS)

   ------------------------------------------------------------------
   -- Wire framing: 4-byte big-endian length prefix + payload.
   ------------------------------------------------------------------

   procedure Write_Frame (T : access Byte_Transport'Class; Payload : Byte_Array)
   is
      Hdr : Byte_Array (0 .. 3);
      L   : constant Natural := Payload'Length;
   begin
      Hdr (0) := U8 (Shift_Right (U32 (L), 24) and 16#FF#);
      Hdr (1) := U8 (Shift_Right (U32 (L), 16) and 16#FF#);
      Hdr (2) := U8 (Shift_Right (U32 (L), 8)  and 16#FF#);
      Hdr (3) := U8 (U32 (L) and 16#FF#);
      Write (T.all, Hdr);
      if L > 0 then
         Write (T.all, Payload);
      end if;
   end Write_Frame;

   function Read_Frame (T : access Byte_Transport'Class) return Byte_Array is
      Hdr : Byte_Array (0 .. 3);
   begin
      Read (T.all, Hdr);
      declare
         L : constant Natural :=
           Natural (Shift_Left (U32 (Hdr (0)), 24)
                    or Shift_Left (U32 (Hdr (1)), 16)
                    or Shift_Left (U32 (Hdr (2)), 8)
                    or U32 (Hdr (3)));
      begin
         if L > Max_Frame then
            raise Handshake_Error with "frame too large";
         end if;
         return R : Byte_Array (0 .. L - 1) do
            if L > 0 then
               Read (T.all, R);
            end if;
         end return;
      end;
   end Read_Frame;

   ------------------------------------------------------------------
   -- Key schedule shared by both roles.
   ------------------------------------------------------------------

   procedure Derive
     (S_Pub, E_Pub, F_Pub : Key32;
      ES, EE              : Key32;
      K_C2S, K_S2C        : out Key32;
      Transcript          : out Crypto.SHA256.Digest)
   is
      Trans_In : Byte_Array (0 .. Prologue'Length + 96 - 1);
      IKM      : Byte_Array (0 .. 63);
      PRK      : Crypto.SHA256.Digest;
      K        : Byte_Array (0 .. 63);
      P        : Natural := 0;
   begin
      for I in Prologue'Range loop
         Trans_In (P) := U8 (Character'Pos (Prologue (I))); P := P + 1;
      end loop;
      for I in 0 .. 31 loop Trans_In (P) := S_Pub (I); P := P + 1; end loop;
      for I in 0 .. 31 loop Trans_In (P) := E_Pub (I); P := P + 1; end loop;
      for I in 0 .. 31 loop Trans_In (P) := F_Pub (I); P := P + 1; end loop;

      Transcript := Crypto.SHA256.Hash (Trans_In);

      for I in 0 .. 31 loop IKM (I) := ES (I); IKM (32 + I) := EE (I); end loop;

      Crypto.HKDF.Extract (Byte_Array (Transcript), IKM, PRK);
      declare
         Info_B : Byte_Array (0 .. Info'Length - 1);
      begin
         for I in Info'Range loop
            Info_B (I - Info'First) := U8 (Character'Pos (Info (I)));
         end loop;
         Crypto.HKDF.Expand (PRK, Info_B, K);
      end;
      for I in 0 .. 31 loop K_C2S (I) := K (I); K_S2C (I) := K (32 + I); end loop;

      --  Scrub the derivation intermediates (the DH secrets and PRK).
      Wipe (K);
      Wipe (IKM);
      Wipe (Byte_Array (PRK));
   end Derive;

   function Nonce (Counter : U64) return Crypto.AEAD.Nonce_96 is
      N : Crypto.AEAD.Nonce_96 := [others => 0];
   begin
      for I in 0 .. 7 loop
         N (I) := U8 (Shift_Right (Counter, 8 * I) and 16#FF#);
      end loop;
      return N;
   end Nonce;

   Empty : constant Byte_Array (1 .. 0) := [others => 0];

   ------------------------------------------------------------------
   -- Handshakes.
   ------------------------------------------------------------------

   procedure Server_Handshake
     (Ch : out Channel; T : access Byte_Transport'Class; Static_Secret : Key32)
   is
      S_Pub  : constant Key32 := Crypto.X25519.Public_Key (Static_Secret);
      F_Priv : Key32;
      K_C2S, K_S2C : Key32;
      Transcript   : Crypto.SHA256.Digest;
   begin
      Crypto.Random.Fill (F_Priv);
      declare
         E_Pub  : constant Byte_Array := Read_Frame (T);            -- client e
         F_Pub  : constant Key32 := Crypto.X25519.Public_Key (F_Priv);
      begin
         if E_Pub'Length /= 32 then
            raise Handshake_Error with "bad ephemeral length";
         end if;
         Write_Frame (T, F_Pub);                                    -- server f
         declare
            E  : Key32;
            ES : Key32;
            EE : Key32;
            Conf_CT  : Byte_Array (1 .. 0);
            Conf_Tag : Crypto.AEAD.Tag_128;
         begin
            for I in 0 .. 31 loop E (I) := E_Pub (E_Pub'First + I); end loop;
            ES := Crypto.X25519.Scalar_Mult (Static_Secret, E);
            EE := Crypto.X25519.Scalar_Mult (F_Priv, E);
            Derive (S_Pub, E, F_Pub, ES, EE, K_C2S, K_S2C, Transcript);
            Wipe (ES); Wipe (EE);

            --  Key-confirmation: tag over the transcript with K_s2c, nonce 0.
            Crypto.AEAD.Seal (K_S2C, Nonce (0), Byte_Array (Transcript), Empty,
                              Conf_CT, Conf_Tag);
            Write_Frame (T, Conf_Tag);

            Ch.K_Send := K_S2C; Ch.K_Recv := K_C2S;
            Ch.N_Send := 1;     -- nonce 0 consumed by the confirmation
            Ch.N_Recv := 0;
            Ch.Ready := True;
            Wipe (F_Priv); Wipe (K_C2S); Wipe (K_S2C);
         end;
      end;
   end Server_Handshake;

   procedure Client_Handshake
     (Ch : out Channel; T : access Byte_Transport'Class; Server_Public : Key32)
   is
      E_Priv : Key32;
      K_C2S, K_S2C : Key32;
      Transcript   : Crypto.SHA256.Digest;
   begin
      Crypto.Random.Fill (E_Priv);
      declare
         E_Pub : constant Key32 := Crypto.X25519.Public_Key (E_Priv);
      begin
         Write_Frame (T, E_Pub);                                    -- client e
         declare
            F_Frame : constant Byte_Array := Read_Frame (T);        -- server f
            F_Pub   : Key32;
         begin
            if F_Frame'Length /= 32 then
               raise Handshake_Error with "bad ephemeral length";
            end if;
            for I in 0 .. 31 loop F_Pub (I) := F_Frame (F_Frame'First + I); end loop;
            declare
               ES : Key32 := Crypto.X25519.Scalar_Mult (E_Priv, Server_Public);
               EE : Key32 := Crypto.X25519.Scalar_Mult (E_Priv, F_Pub);
            begin
               Derive (Server_Public, E_Pub, F_Pub, ES, EE,
                       K_C2S, K_S2C, Transcript);
               Wipe (ES); Wipe (EE);

               --  Verify the server's key-confirmation tag.
               declare
                  Tag_Frame : constant Byte_Array := Read_Frame (T);
                  Tag : Crypto.AEAD.Tag_128;
                  Got : Byte_Array (1 .. 0);
               begin
                  if Tag_Frame'Length /= 16 then
                     raise Handshake_Error with "bad confirmation length";
                  end if;
                  for I in 0 .. 15 loop
                     Tag (I) := Tag_Frame (Tag_Frame'First + I);
                  end loop;
                  if not Crypto.AEAD.Open (K_S2C, Nonce (0),
                        Byte_Array (Transcript), Empty, Tag, Got)
                  then
                     raise Auth_Error with "server authentication failed";
                  end if;
               end;

               Ch.K_Send := K_C2S; Ch.K_Recv := K_S2C;
               Ch.N_Send := 0;
               Ch.N_Recv := 1;     -- nonce 0 consumed by the confirmation
               Ch.Ready := True;
               Wipe (E_Priv); Wipe (K_C2S); Wipe (K_S2C);
            end;
         end;
      end;
   end Client_Handshake;

   ------------------------------------------------------------------
   -- Records.
   ------------------------------------------------------------------

   procedure Send_Message
     (Ch : in out Channel; T : access Byte_Transport'Class;
      Plaintext : Byte_Array)
   is
      Frame : Byte_Array (0 .. Plaintext'Length + 16 - 1);
      CT    : Byte_Array (0 .. Plaintext'Length - 1);
      Tag   : Crypto.AEAD.Tag_128;
   begin
      Crypto.AEAD.Seal (Ch.K_Send, Nonce (Ch.N_Send), Empty, Plaintext, CT, Tag);
      for I in CT'Range loop Frame (I) := CT (I); end loop;
      for I in 0 .. 15 loop Frame (Plaintext'Length + I) := Tag (I); end loop;
      Write_Frame (T, Frame);
      Ch.N_Send := Ch.N_Send + 1;
   end Send_Message;

   function Recv_Message
     (Ch : in out Channel; T : access Byte_Transport'Class) return Byte_Array
   is
      Frame : constant Byte_Array := Read_Frame (T);
   begin
      if Frame'Length < 16 then
         raise Auth_Error with "short frame";
      end if;
      declare
         PT_Len : constant Natural := Frame'Length - 16;
         CT  : Byte_Array (0 .. PT_Len - 1);
         Tag : Crypto.AEAD.Tag_128;
         PT  : Byte_Array (0 .. PT_Len - 1);
      begin
         for I in 0 .. PT_Len - 1 loop CT (I) := Frame (Frame'First + I); end loop;
         for I in 0 .. 15 loop Tag (I) := Frame (Frame'First + PT_Len + I); end loop;
         if not Crypto.AEAD.Open (Ch.K_Recv, Nonce (Ch.N_Recv), Empty, CT, Tag, PT)
         then
            raise Auth_Error with "record authentication failed";
         end if;
         Ch.N_Recv := Ch.N_Recv + 1;
         return PT;
      end;
   end Recv_Message;

   procedure Close (Ch : in out Channel) is
   begin
      Wipe (Ch.K_Send);
      Wipe (Ch.K_Recv);
      Ch.N_Send := 0;
      Ch.N_Recv := 0;
      Ch.Ready := False;
   end Close;

end Secure_Channel;
