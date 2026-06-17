---------------------------------------------------------------------
-- Protocol — application record tags carried inside Secure_Channel frames.
--
-- Each decrypted record starts with a one-byte tag. Client->server sends a
-- Prompt record; server->client streams Prefill ticks and Token pieces,
-- terminated by a Done record.
---------------------------------------------------------------------

with Crypto;

package Protocol is

   Tag_Session : constant Crypto.U8 := Character'Pos ('s');  -- both: session id
                                                            --   (C->S desired / empty = new; S->C assigned)
   Tag_Prompt  : constant Crypto.U8 := Character'Pos ('p');  -- C->S: user text
   Tag_Token   : constant Crypto.U8 := Character'Pos ('t');  -- S->C: a token piece
   Tag_Prefill : constant Crypto.U8 := Character'Pos ('.');  -- S->C: prefill tick
   Tag_Done    : constant Crypto.U8 := Character'Pos ('!');  -- S->C: end of reply
   Tag_Error   : constant Crypto.U8 := Character'Pos ('e');  -- S->C: error + reason

   --  OpenAI-compatible API, tunneled over the same encrypted channel. The
   --  local proxy translates HTTP <-> these records.
   Tag_Chat    : constant Crypto.U8 := Character'Pos ('q');  -- C->S: /v1/chat/completions JSON body
   Tag_Models  : constant Crypto.U8 := Character'Pos ('m');  -- C->S: list models (no body)
   Tag_Resp    : constant Crypto.U8 := Character'Pos ('r');  -- S->C: a JSON response (non-stream)
   --  Streaming chat reuses Tag_Token (each piece = a chat.completion.chunk
   --  delta) terminated by Tag_Done.

end Protocol;
