---------------------------------------------------------------------
-- Crypto.ChaCha20 body — RFC 8439 §2.1–2.4
---------------------------------------------------------------------

with Interfaces; use Interfaces;

package body Crypto.ChaCha20 is

   type State is array (0 .. 15) of U32;

   --  "expand 32-byte k"
   C0 : constant U32 := 16#61707865#;
   C1 : constant U32 := 16#3320646E#;
   C2 : constant U32 := 16#79622D32#;
   C3 : constant U32 := 16#6B206574#;

   procedure Quarter_Round (S : in out State; A, B, C, D : Natural) is
   begin
      S (A) := S (A) + S (B); S (D) := Rotate_Left (S (D) xor S (A), 16);
      S (C) := S (C) + S (D); S (B) := Rotate_Left (S (B) xor S (C), 12);
      S (A) := S (A) + S (B); S (D) := Rotate_Left (S (D) xor S (A), 8);
      S (C) := S (C) + S (D); S (B) := Rotate_Left (S (B) xor S (C), 7);
   end Quarter_Round;

   procedure Keystream_Block
     (Key : Key_256; Nonce : Nonce_96; Counter : U32; B : out Block_64)
   is
      Init, S : State;
   begin
      Init (0) := C0;  Init (1) := C1;  Init (2) := C2;  Init (3) := C3;
      for I in 0 .. 7 loop
         Init (4 + I) := Load_LE32 (Key, 4 * I);
      end loop;
      Init (12) := Counter;
      for I in 0 .. 2 loop
         Init (13 + I) := Load_LE32 (Nonce, 4 * I);
      end loop;

      S := Init;
      for Double_Round in 1 .. 10 loop
         --  column rounds
         Quarter_Round (S, 0, 4,  8, 12);
         Quarter_Round (S, 1, 5,  9, 13);
         Quarter_Round (S, 2, 6, 10, 14);
         Quarter_Round (S, 3, 7, 11, 15);
         --  diagonal rounds
         Quarter_Round (S, 0, 5, 10, 15);
         Quarter_Round (S, 1, 6, 11, 12);
         Quarter_Round (S, 2, 7,  8, 13);
         Quarter_Round (S, 3, 4,  9, 14);
      end loop;

      for I in 0 .. 15 loop
         Store_LE32 (B, 4 * I, S (I) + Init (I));
      end loop;
   end Keystream_Block;

   procedure XOR_Stream
     (Key     : Key_256;
      Nonce   : Nonce_96;
      Counter : U32;
      Input   : Byte_Array;
      Output  : out Byte_Array)
   is
      KS     : Block_64;
      N      : constant Natural := Input'Length;
      Blocks : constant Natural := (N + 63) / 64;
      Pos    : Natural := 0;
   begin
      for Blk in 0 .. Blocks - 1 loop
         Keystream_Block (Key, Nonce, Counter + U32 (Blk), KS);
         declare
            This : constant Natural := Natural'Min (64, N - Pos);
         begin
            for I in 0 .. This - 1 loop
               Output (Output'First + Pos + I) :=
                 Input (Input'First + Pos + I) xor KS (I);
            end loop;
            Pos := Pos + This;
         end;
      end loop;
   end XOR_Stream;

end Crypto.ChaCha20;
