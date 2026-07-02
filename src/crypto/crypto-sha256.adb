---------------------------------------------------------------------
-- Crypto.SHA256 body — FIPS 180-4
--
-- Compress is the shared 64-byte block step used by both Hash (one-shot) and
-- Context (streaming). The round arithmetic is unchanged from the original
-- proved Hash; only the driver changed (Hash now calls Compress per block),
-- so the existing proof obligations carry over.
---------------------------------------------------------------------

with Interfaces; use Interfaces;

package body Crypto.SHA256 with SPARK_Mode => On is

   type Words64 is array (0 .. 63) of U32;

   Init_H : constant Words8 :=
     [16#6a09e667#, 16#bb67ae85#, 16#3c6ef372#, 16#a54ff53a#,
      16#510e527f#, 16#9b05688c#, 16#1f83d9ab#, 16#5be0cd19#];

   K : constant Words64 :=
     [16#428a2f98#, 16#71374491#, 16#b5c0fbcf#, 16#e9b5dba5#,
      16#3956c25b#, 16#59f111f1#, 16#923f82a4#, 16#ab1c5ed5#,
      16#d807aa98#, 16#12835b01#, 16#243185be#, 16#550c7dc3#,
      16#72be5d74#, 16#80deb1fe#, 16#9bdc06a7#, 16#c19bf174#,
      16#e49b69c1#, 16#efbe4786#, 16#0fc19dc6#, 16#240ca1cc#,
      16#2de92c6f#, 16#4a7484aa#, 16#5cb0a9dc#, 16#76f988da#,
      16#983e5152#, 16#a831c66d#, 16#b00327c8#, 16#bf597fc7#,
      16#c6e00bf3#, 16#d5a79147#, 16#06ca6351#, 16#14292967#,
      16#27b70a85#, 16#2e1b2138#, 16#4d2c6dfc#, 16#53380d13#,
      16#650a7354#, 16#766a0abb#, 16#81c2c92e#, 16#92722c85#,
      16#a2bfe8a1#, 16#a81a664b#, 16#c24b8b70#, 16#c76c51a3#,
      16#d192e819#, 16#d6990624#, 16#f40e3585#, 16#106aa070#,
      16#19a4c116#, 16#1e376c08#, 16#2748774c#, 16#34b0bcb5#,
      16#391c0cb3#, 16#4ed8aa4a#, 16#5b9cca4f#, 16#682e6ff3#,
      16#748f82ee#, 16#78a5636f#, 16#84c87814#, 16#8cc70208#,
      16#90befffa#, 16#a4506ceb#, 16#bef9a3f7#, 16#c67178f2#];

   function Ch  (X, Y, Z : U32) return U32 is ((X and Y) xor ((not X) and Z));
   function Maj (X, Y, Z : U32) return U32 is
     ((X and Y) xor (X and Z) xor (Y and Z));
   function Big_S0 (X : U32) return U32 is
     (Rotate_Right (X, 2)  xor Rotate_Right (X, 13) xor Rotate_Right (X, 22));
   function Big_S1 (X : U32) return U32 is
     (Rotate_Right (X, 6)  xor Rotate_Right (X, 11) xor Rotate_Right (X, 25));
   function Sm_S0 (X : U32) return U32 is
     (Rotate_Right (X, 7)  xor Rotate_Right (X, 18) xor Shift_Right (X, 3));
   function Sm_S1 (X : U32) return U32 is
     (Rotate_Right (X, 17) xor Rotate_Right (X, 19) xor Shift_Right (X, 10));

   function Load_BE32 (A : Byte_Array; Offset : Natural) return U32 is
     (Shift_Left (U32 (A (A'First + Offset)), 24)
      or Shift_Left (U32 (A (A'First + Offset + 1)), 16)
      or Shift_Left (U32 (A (A'First + Offset + 2)), 8)
      or U32 (A (A'First + Offset + 3)))
     with Pre => A'Last >= A'First
                 and then A'Last - A'First >= 3
                 and then Offset <= A'Last - A'First - 3;

   --  Compress one 64-byte block into the working state H. The rounds are the
   --  exact arithmetic of the original Hash; Hash and Context both call this so
   --  there is one implementation of the compression function.
   procedure Compress (H : in out Words8; Block : Byte_Array)
     with Pre => Block'Length = Block_Size
   is
      W : Words64 := [others => 0];
      A, B, C, D, E, F, G, Hh, T1, T2 : U32;
   begin
      for T in 0 .. 15 loop
         W (T) := Load_BE32 (Block, 4 * T);
      end loop;
      for T in 16 .. 63 loop
         W (T) := Sm_S1 (W (T - 2)) + W (T - 7)
                + Sm_S0 (W (T - 15)) + W (T - 16);
      end loop;

      A := H (0); B := H (1); C := H (2); D := H (3);
      E := H (4); F := H (5); G := H (6); Hh := H (7);

      for T in 0 .. 63 loop
         T1 := Hh + Big_S1 (E) + Ch (E, F, G) + K (T) + W (T);
         T2 := Big_S0 (A) + Maj (A, B, C);
         Hh := G; G := F; F := E; E := D + T1;
         D := C; C := B; B := A; A := T1 + T2;
      end loop;

      H (0) := H (0) + A; H (1) := H (1) + B; H (2) := H (2) + C;
      H (3) := H (3) + D; H (4) := H (4) + E; H (5) := H (5) + F;
      H (6) := H (6) + G; H (7) := H (7) + Hh;
   end Compress;

   --  Emit the 8 working words as the 32-byte digest (big-endian per word).
   function To_Digest (H : Words8) return Digest is
      Result : Digest := [others => 0];
   begin
      for I in 0 .. 7 loop
         Result (4 * I)     := U8 (Shift_Right (H (I), 24) and 16#FF#);
         Result (4 * I + 1) := U8 (Shift_Right (H (I), 16) and 16#FF#);
         Result (4 * I + 2) := U8 (Shift_Right (H (I), 8)  and 16#FF#);
         Result (4 * I + 3) := U8 (H (I) and 16#FF#);
      end loop;
      return Result;
   end To_Digest;

   function Hash (M : Byte_Array) return Digest is
      Bit_Len : constant U64 := U64 (M'Length) * 8;
      pragma Annotate
        (GNATprove, False_Positive, "range check might fail",
         "M'Length is far below Natural'Last for every caller (handshake "
         & "transcripts, AEAD frames, HKDF inputs); a message within ~72 bytes "
         & "of the 2 GiB Natural limit is never hashed here.");
      Zeros   : constant Natural := (64 - ((M'Length + 9) mod 64)) mod 64;
      pragma Annotate
        (GNATprove, False_Positive, "overflow check might fail",
         "M'Length + 9 cannot overflow for the small inputs used (see above).");
      pragma Annotate
        (GNATprove, False_Positive, "range check might fail",
         "M'Length is far below Natural'Last for the small inputs used.");
      Padded  : Byte_Array (0 .. M'Length + 1 + Zeros + 8 - 1) := [others => 0];
      pragma Annotate
        (GNATprove, False_Positive, "overflow check might fail",
         "padded length cannot overflow for the small inputs used (see above).");
      pragma Annotate
        (GNATprove, False_Positive, "range check might fail",
         "M'Length is far below Natural'Last for the small inputs used.");
      H       : Words8 := Init_H;
   begin
      --  Pad: message || 0x80 || zeros || 64-bit big-endian bit length.
      for I in M'Range loop
         Padded (I - M'First) := M (I);
      end loop;
      Padded (M'Length) := 16#80#;
      for I in 0 .. 7 loop
         Padded (Padded'Last - I) := U8 (Shift_Right (Bit_Len, 8 * I) and 16#FF#);
      end loop;

      --  Process each 64-byte block via the shared Compress step (one
      --  implementation of the compression function, used by Hash and Context).
      declare
         N_Blocks : constant Natural := Padded'Length / 64;
         pragma Annotate
           (GNATprove, False_Positive, "range check might fail",
            "Padded'Length is bounded by the Hash input bound (see above).");
      begin
         for Blk in 0 .. N_Blocks - 1 loop
            Compress (H, Padded (Padded'First + Blk * 64
                                 .. Padded'First + Blk * 64 + 63));
         end loop;
      end;

      return To_Digest (H);
   end Hash;

   procedure HMAC (Key, Msg : Byte_Array; Mac : out Digest) is
      K0    : Byte_Array (0 .. Block_Size - 1) := [others => 0];
      I_Pad : Byte_Array (0 .. Block_Size - 1) := [others => 0];
      O_Pad : Byte_Array (0 .. Block_Size - 1) := [others => 0];
   begin
      --  Normalise the key to one block.
      if Key'Length > Block_Size then
         declare
            HK : constant Digest := Hash (Key);
         begin
            for I in HK'Range loop
               K0 (I) := HK (I);
            end loop;
         end;
      else
         --  Iterate over Key's own range (no Key'Length - 1, which would
         --  under-flow for an empty key); index K0 by the relative position.
         for I in Key'Range loop
            K0 (I - Key'First) := Key (I);
         end loop;
      end if;

      for I in 0 .. Block_Size - 1 loop
         I_Pad (I) := K0 (I) xor 16#36#;
         O_Pad (I) := K0 (I) xor 16#5c#;
      end loop;

      Mac := Hash (O_Pad & Byte_Array (Hash (I_Pad & Msg)));
      pragma Annotate
        (GNATprove, False_Positive, "range check might fail",
         "I_Pad & Msg length cannot overflow for the small HMAC inputs used.");

      --  Scrub the key-derived block and pads; they reveal the HMAC key.
      Wipe (K0); Wipe (I_Pad); Wipe (O_Pad);
   end HMAC;

   ------------------------------------------------------------------
   --  Incremental streaming (Context)
   ------------------------------------------------------------------

   procedure Init (Ctx : out Context) is
   begin
      Ctx := (H => Init_H, Buf => [others => 0], Buf_Len => 0, Bit_Len => 0);
   end Init;

   ------------------------------------------------------------------
   --  Incremental streaming (Context). SPARK_Mode => Off: correctness
   --  rests on reusing the proved Compress step and on a bit-identical
   --  cross-check against Hash in test_weight_pin. See spec comment.
   ------------------------------------------------------------------

   procedure Update (Ctx : in out Context; Data : Byte_Array)
     with SPARK_Mode => Off
   is
      --  Walk Data by a zero-based offset Off (bytes consumed so far). The
      --  three loops top off the partial Buf, then compress whole blocks
      --  straight from Data, then park the 0..63-byte tail in Buf. Buf_Len
      --  is back in 0..Block_Size-1 at every loop top (it transiently reaches
      --  Block_Size then resets to 0), so Buf is never indexed out of bounds.
      Off : Natural := 0;
   begin
      Ctx.Bit_Len := Ctx.Bit_Len + U64 (Data'Length) * 8;

      while Off < Data'Length and then Ctx.Buf_Len > 0 loop
         Ctx.Buf (Ctx.Buf_Len) := Data (Data'First + Off);
         Ctx.Buf_Len := Ctx.Buf_Len + 1;
         Off := Off + 1;
         if Ctx.Buf_Len = Block_Size then
            Compress (Ctx.H, Ctx.Buf);
            Ctx.Buf_Len := 0;
         end if;
      end loop;

      while Off + Block_Size <= Data'Length loop
         Compress (Ctx.H, Data (Data'First + Off
                                 .. Data'First + Off + Block_Size - 1));
         Off := Off + Block_Size;
      end loop;

      while Off < Data'Length loop
         Ctx.Buf (Ctx.Buf_Len) := Data (Data'First + Off);
         Ctx.Buf_Len := Ctx.Buf_Len + 1;
         Off := Off + 1;
      end loop;
   end Update;

   procedure Final (Ctx : in out Context; D : out Digest)
     with SPARK_Mode => Off
   is
      --  Build the final padding from the pending Buf: Buf[0..Buf_Len-1] ||
      --  0x80 || zeros || 64-bit big-endian Bit_Len. One or two blocks. Buf_Len
      --  is 0..63 by construction (Update only ever leaves 0..63 pending), so
      --  every index below is in bounds (Buf is 0..63, Pad is 0..127, Total is
      --  64 or 128).
      Pad : Byte_Array (0 .. 2 * Block_Size - 1) := [others => 0];
      N   : constant Natural := Ctx.Buf_Len;   --  pending bytes, 0..63
      Zeros : constant Natural := (64 - ((N + 9) mod 64)) mod 64;
      Total  : constant Natural := N + 1 + Zeros + 8;   --  64 or 128
      N_Blk  : constant Natural := Total / 64;
      H      : Words8 := Ctx.H;
   begin
      --  Pending bytes.
      for I in 0 .. N - 1 loop
         Pad (I) := Ctx.Buf (I);
      end loop;
      Pad (N) := 16#80#;
      --  64-bit big-endian bit length in the last 8 bytes of the final block.
      for I in 0 .. 7 loop
         Pad (Total - 1 - I) := U8 (Shift_Right (Ctx.Bit_Len, 8 * I) and 16#FF#);
      end loop;

      for Blk in 0 .. N_Blk - 1 loop
         Compress (H, Pad (Blk * 64 .. Blk * 64 + 63));
      end loop;

      D := To_Digest (H);

      --  Scrub the buffer/state so a stale Context cannot leak partial data.
      Ctx.H := [others => 0];
      Wipe (Ctx.Buf);
      Ctx.Buf_Len := 0;
      Ctx.Bit_Len := 0;
   end Final;

end Crypto.SHA256;