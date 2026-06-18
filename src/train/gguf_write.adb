---------------------------------------------------------------------
-- GGUF_Write body — explicit little-endian serialization.
---------------------------------------------------------------------

with Interfaces;            use Interfaces;
with Ada.Unchecked_Conversion;
with Ada.Streams.Stream_IO;
with Ada.Streams;           use Ada.Streams;

package body GGUF_Write is

   --  GGUF metadata value-type tags (match LLM_GGUF.GGUF_Value_Type).
   T_U32 : constant := 4;
   T_F32 : constant := 6;
   T_STR : constant := 8;
   T_ARR : constant := 9;

   function To_U32 is new Ada.Unchecked_Conversion (Float, Unsigned_32);

   --------------------------------------------------------------------
   --  Little-endian byte emitters (each char of the result == one byte)
   --------------------------------------------------------------------
   function B_U32 (V : Unsigned_32) return String is
      S : String (1 .. 4);
      X : Unsigned_32 := V;
   begin
      for I in 1 .. 4 loop
         S (I) := Character'Val (Natural (X and 16#FF#));
         X := Shift_Right (X, 8);
      end loop;
      return S;
   end B_U32;

   function B_U64 (V : Unsigned_64) return String is
      S : String (1 .. 8);
      X : Unsigned_64 := V;
   begin
      for I in 1 .. 8 loop
         S (I) := Character'Val (Natural (X and 16#FF#));
         X := Shift_Right (X, 8);
      end loop;
      return S;
   end B_U64;

   function B_F32 (V : Float) return String is (B_U32 (To_U32 (V)));

   function B_Str (S : String) return String is   -- u64 length + bytes
     (B_U64 (Unsigned_64 (S'Length)) & S);

   procedure Add_Meta (B : in out Builder; Bytes : String) is
   begin
      Append (B.Meta.Bytes, Bytes);
      B.Meta.Count := B.Meta.Count + 1;
   end Add_Meta;

   --------------------------------------------------------------------
   --  Metadata
   --------------------------------------------------------------------
   procedure Meta_Str (B : in out Builder; Key, Val : String) is
   begin
      Add_Meta (B, B_Str (Key) & B_U32 (T_STR) & B_Str (Val));
   end Meta_Str;

   procedure Meta_U32 (B : in out Builder; Key : String; Val : Natural) is
   begin
      Add_Meta (B, B_Str (Key) & B_U32 (T_U32) & B_U32 (Unsigned_32 (Val)));
   end Meta_U32;

   procedure Meta_F32 (B : in out Builder; Key : String; Val : Float) is
   begin
      Add_Meta (B, B_Str (Key) & B_U32 (T_F32) & B_F32 (Val));
   end Meta_F32;

   procedure Meta_Str_Array (B : in out Builder; Key : String; Vals : Str_List) is
      Body_S : Unbounded_String;
   begin
      Append (Body_S, B_Str (Key));
      Append (Body_S, B_U32 (T_ARR));
      Append (Body_S, B_U32 (T_STR));                       -- array element type
      Append (Body_S, B_U64 (Unsigned_64 (Vals'Length)));   -- array length
      for V of Vals loop
         Append (Body_S, B_Str (To_String (V)));
      end loop;
      Add_Meta (B, To_String (Body_S));
   end Meta_Str_Array;

   --------------------------------------------------------------------
   --  Tensors
   --------------------------------------------------------------------
   procedure Add_Tensor_F32
     (B : in out Builder; Name : String; Dims : Dims_Array; Data : Float_Array)
   is
      T : Tensor_Rec;
   begin
      T.Name   := To_Unbounded_String (Name);
      T.N_Dims := Dims'Length;
      for I in 1 .. Dims'Length loop
         T.Dims (I) := Dims (Dims'First + I - 1);
      end loop;
      for E of Data loop
         Append (T.Data, B_F32 (E));
      end loop;
      B.Tens.Append (T);
   end Add_Tensor_F32;

   --------------------------------------------------------------------
   --  Save
   --------------------------------------------------------------------
   Align : constant := 32;
   function Round_Up (N : Natural) return Natural is
     (((N + Align - 1) / Align) * Align);

   procedure Save (B : in out Builder; Path : String) is
      use Ada.Streams.Stream_IO;
      F   : File_Type;
      Hdr : Unbounded_String;
      TI  : Unbounded_String;   -- tensor infos
      Off : Natural := 0;       -- running data offset (relative to data start)

      procedure Put (Str : String) is
         Buf : Stream_Element_Array (1 .. Stream_Element_Offset (Str'Length));
      begin
         for I in Str'Range loop
            Buf (Stream_Element_Offset (I - Str'First + 1)) :=
              Stream_Element (Character'Pos (Str (I)));
         end loop;
         Write (F, Buf);
      end Put;

      procedure Pad (N : Natural) is
      begin
         if N > 0 then Put ([1 .. N => Character'Val (0)]); end if;
      end Pad;
   begin
      --  header
      Append (Hdr, "GGUF");
      Append (Hdr, B_U32 (3));
      Append (Hdr, B_U64 (Unsigned_64 (B.Tens.Length)));
      Append (Hdr, B_U64 (Unsigned_64 (B.Meta.Count)));

      --  tensor infos (assign aligned offsets)
      for T of B.Tens loop
         Append (TI, B_Str (To_String (T.Name)));
         Append (TI, B_U32 (Unsigned_32 (T.N_Dims)));
         for D in 1 .. T.N_Dims loop
            Append (TI, B_U64 (Unsigned_64 (T.Dims (D))));
         end loop;
         Append (TI, B_U32 (0));                      -- GGML type 0 = F32
         Append (TI, B_U64 (Unsigned_64 (Off)));
         Off := Round_Up (Off + Length (T.Data));
      end loop;

      Create (F, Out_File, Path);
      Put (To_String (Hdr));
      Put (To_String (B.Meta.Bytes));
      Put (To_String (TI));

      --  pad to the aligned data section
      declare
         Pos        : constant Natural := Length (Hdr) + Length (B.Meta.Bytes) + Length (TI);
         Data_Start : constant Natural := Round_Up (Pos);
         Cursor     : Natural := 0;       -- bytes written within the data section
      begin
         Pad (Data_Start - Pos);
         for T of B.Tens loop
            --  each tensor's data begins at its (aligned) offset
            declare
               Want : constant Natural := Round_Up (Cursor);
            begin
               Pad (Want - Cursor);
               Cursor := Want;
            end;
            Put (To_String (T.Data));
            Cursor := Cursor + Length (T.Data);
         end loop;
      end;
      Close (F);
   end Save;

end GGUF_Write;
