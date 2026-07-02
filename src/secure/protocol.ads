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

   --  Optional client authentication (opt-in). When the server is started with
   --  ASPIDA_CLIENT_TOKEN set, a client must send this as its FIRST record,
   --  body = the shared token, before the session hello. A client that has the
   --  token configured always sends it first; a server without a token simply
   --  consumes and ignores a leading Tag_Auth, so the two sides interoperate
   --  in every on/off combination.
   Tag_Auth    : constant Crypto.U8 := Character'Pos ('a');  -- C->S: shared token

   --  OpenAI-compatible API, tunneled over the same encrypted channel. The
   --  local proxy translates HTTP <-> these records.
   Tag_Chat    : constant Crypto.U8 := Character'Pos ('q');  -- C->S: /v1/chat/completions JSON body
   Tag_Models  : constant Crypto.U8 := Character'Pos ('m');  -- C->S: list models (no body)
   Tag_Resp    : constant Crypto.U8 := Character'Pos ('r');  -- S->C: a JSON response (non-stream)

   --  Model selection: client asks the server to make a discovered model the
   --  active one. Body = the model id (its absolute path, as returned in the
   --  models list). The server validates it against the catalog, persists the
   --  choice, and replies Tag_Resp with a JSON {ok, reload, message}. When a
   --  supervisor is present (ASPIDA_AUTORELOAD) the server then reloads.
   Tag_Select  : constant Crypto.U8 := Character'Pos ('M');  -- C->S: select active model
   --  Streaming chat reuses Tag_Token (each piece = a chat.completion.chunk
   --  delta) terminated by Tag_Done.

   --  Chat event sub-channels: emitted in-stream between Tag_Token text
   --  pieces so the proxy can render reasoning_content / tool_calls without
   --  trying to second-guess the chat-template parser. All are one-byte tag +
   --  body; Tag_Finish_Reason and Tag_Tool_Call carry the full structured
   --  payload (no incremental deltas needed — we only emit one Tag_Tool_Call
   --  per assembled tool invocation).
   Tag_Reasoning_Begin : constant Crypto.U8 := Character'Pos ('R'); -- S->C: opening a <think> block
   Tag_Text_Begin      : constant Crypto.U8 := Character'Pos ('Z'); -- S->C: assistant text begins (after thinking/tools)
   Tag_Tool_Call       : constant Crypto.U8 := Character'Pos ('T'); -- S->C: body = JSON {"id","name","arguments"}
   Tag_Finish_Reason   : constant Crypto.U8 := Character'Pos ('F'); -- S->C: "stop" / "length" / "tool_calls"

   --  H19 weight-streaming (docs/H19_WEIGHT_STREAM_ROADMAP.md). A thin mode of
   --  the server serves encrypted byte-ranges of a model file; the client's
   --  Remote_AEAD_Source fetches them through this same channel. The server
   --  holds no secret (no prompt/activation ever transits or resides there);
   --  the channel's AEAD still defeats a network MITM and tamper with the
   --  weight bytes. Integrity of the *artifact* is a separate Phase 4 concern
   --  (signed manifest, ASPIDA_WEIGHT_PUBKEY); these records only carry bytes.
   --
   --  Request body  (C->S, Tag_WReq): Offset(U64 LE, 8) + Count(U32 LE, 4)
   --                                  + Model_ID_Len(U32 LE, 4) + Model_ID
   --  Response body (S->C, Tag_WData): exactly Count bytes (the length is
   --                                  implicit from the record length minus 1)
   --  Error body    (S->C, Tag_WErr): a UTF-8 reason string
   Tag_WReq  : constant Crypto.U8 := Character'Pos ('W');  -- C->S: fetch a byte range
   Tag_WData : constant Crypto.U8 := Character'Pos ('D');  -- S->C: the requested bytes
   Tag_WErr  : constant Crypto.U8 := Character'Pos ('X');  -- S->C: range error + reason

end Protocol;
