---------------------------------------------------------------------
-- LLM_Weight_Proto — codec for the H19 weight-streaming wire records.
--
-- These records ride inside the existing Secure_Channel AEAD frames (see
-- Protocol: Tag_WReq / Tag_WData / Tag_WErr). The channel hands a plaintext
-- Crypto.Byte_Array to Send_Message / Recv_Message; the first byte is the
-- Protocol tag and the rest is the body. This package encodes and decodes
-- the bodies so neither the client (Remote_AEAD_Source) nor the server
-- (the weight-range responder) repeats the little-endian packing logic.
--
-- Request body  : Offset(U64 LE) + Count(U32 LE) + Model_ID_Len(U32 LE)
--                + Model_ID (UTF-8 bytes).
-- Response body : exactly Count bytes (length implicit from the record).
-- Error body    : a UTF-8 reason string.
--
-- Bounds are checked on decode: a truncated or inconsistent request yields
-- OK = False rather than raising, so the server can reply Tag_WErr instead
-- of dropping the channel. This is Phase 1+2 of H19.
--
-- SPARK_Mode: the egress encoders (Tag_Of / Encode_WReq / Encode_WData /
-- Encode_WErr) are SPARK_Mode => On and flow-analysed by `make prove-egress`
-- (gnatprove --mode=flow). They are the client egress content seam — every
-- byte the client sends to the weight store is a Tag_WReq whose body is a
-- pure function of (Off, Count, Model_ID), so flow analysis confirms no
-- prompt/output data reaches the encoder. Decode_WReq is SPARK_Mode => Off
-- (it builds an Ada.Strings.Unbounded.Unbounded_String from the wire, which
-- is heap-backed and outside SPARK's scope); its correctness rests on the
-- round-trip assertions in test_weight_egress.
---------------------------------------------------------------------

with Ada.Strings.Unbounded;
with Crypto;
with Interfaces;
use type Interfaces.Unsigned_32;   --  '>=' / '<=' on Count (Crypto.U32) in the
                                   --  Encode_WReq precondition
use type Interfaces.Unsigned_64;   --  arithmetic on Offset in the same

package LLM_Weight_Proto with SPARK_Mode => On is

   --  Decode the first byte of a record. Returns 0 for an empty record (no
   --  valid tag is 0, so callers can treat 0 as "no message").
   function Tag_Of (Msg : Crypto.Byte_Array) return Crypto.U8;

   --  Encode a weight-range request. Count must be <= Max_Range_Count.
   function Encode_WReq
     (Off      : Crypto.U64;
      Count    : Crypto.U32;
      Model_ID : String) return Crypto.Byte_Array
     with Pre => Count <= Max_Range_Count;

   --  Encode a weight-data response (tag + the bytes).
   function Encode_WData (Data : Crypto.Byte_Array) return Crypto.Byte_Array;

   --  Encode an error response (tag + reason).
   function Encode_WErr (Reason : String) return Crypto.Byte_Array;

   --  Decode a weight-range request. OK is False if the record is too short
   --  or the declared Model_ID length runs past the end. On OK = False the
   --  other out parameters are set to zero / empty.
   procedure Decode_WReq
     (Msg      : Crypto.Byte_Array;
      Off      : out Crypto.U64;
      Count    : out Crypto.U32;
      Model_ID : out Ada.Strings.Unbounded.Unbounded_String;
      OK       : out Boolean)
     with SPARK_Mode => Off;

   --  Largest byte range a single request may ask for. The channel frame cap
   --  (1 MiB) bounds a response; 64 KiB keeps every request well under it and
   --  matches the Remote_AEAD_Source chunk size.
   Max_Range_Count : constant Crypto.U32 := 65_536;

end LLM_Weight_Proto;