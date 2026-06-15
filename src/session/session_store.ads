---------------------------------------------------------------------
-- Session_Store — encrypted, persistent conversation history.
--
-- Accumulates a session transcript in RAM and, after each turn, persists
-- it to sessions/<id>.session encrypted at rest (At_Rest: ChaCha20-
-- Poly1305 under a PBKDF2 key from the server master password in the
-- ASPIDA_STORE_PASSWORD environment variable). If that variable is unset,
-- persistence is disabled (nothing is written to disk).
---------------------------------------------------------------------

package Session_Store is

   type Store is limited private;

   --  True when ASPIDA_STORE_PASSWORD is set (history is persisted).
   function Enabled return Boolean;

   --  True if Id is safe to use as the on-disk session filename: 1..64 chars
   --  from [A-Za-z0-9_-] only. Rejects anything that could escape the sessions
   --  directory (path separators, '.', NUL), i.e. guards against traversal.
   function Valid_Id (Id : String) return Boolean;

   --  Begin or resume the session with the given id: if an encrypted
   --  transcript already exists it is decrypted and loaded.
   procedure Open (S : out Store; Id : String);

   --  Record one (user, assistant) turn and persist the re-encrypted
   --  transcript. No-op when persistence is disabled.
   procedure Append_Turn (S : in out Store; User, Assistant : String);

   --  Number of recorded turns (including any loaded on resume).
   function Turn_Count (S : Store) return Natural;

   --  The user / assistant text of turn I (1 .. Turn_Count), for rebuilding
   --  multi-turn context.
   function User_Of (S : Store; I : Positive) return String;
   function Assistant_Of (S : Store; I : Positive) return String;

   --  A readable concatenation of all turns (display / diagnostics).
   function Transcript (S : Store) return String;

   --  Release in-RAM transcript/credentials.
   procedure Close (S : in out Store);

private

   type Store_Rec;
   type Store is access Store_Rec;

end Session_Store;
