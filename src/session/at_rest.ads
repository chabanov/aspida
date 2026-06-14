---------------------------------------------------------------------
-- At_Rest — password-protected encrypted file storage for session data
-- (conversation history, KV state, ...). ChaCha20-Poly1305 under a key
-- derived from the password with PBKDF2; the random salt, nonce and
-- iteration count are stored in (and authenticated as part of) the file,
-- so Load needs only the password. A wrong password or any tampering is
-- rejected (Decrypt_Error), never silently accepted.
---------------------------------------------------------------------

with Crypto;

package At_Rest is

   Decrypt_Error : exception;
   Format_Error  : exception;

   procedure Save
     (Path       : String;
      Password   : Crypto.Byte_Array;
      Plaintext  : Crypto.Byte_Array;
      Iterations : Positive := 200_000);

   function Load
     (Path : String; Password : Crypto.Byte_Array) return Crypto.Byte_Array;

end At_Rest;
