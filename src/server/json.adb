---------------------------------------------------------------------
-- JSON body — recursive-descent parser, builders, compact serializer.
---------------------------------------------------------------------

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body JSON is

   LF : constant Character := Character'Val (10);
   CR : constant Character := Character'Val (13);
   HT : constant Character := Character'Val (9);

   ----------------------------------------------------------------
   -- Builders
   ----------------------------------------------------------------
   function New_Object return Value_Ref is (new Value'(Kind => J_Object, others => <>));
   function New_Array  return Value_Ref is (new Value'(Kind => J_Array,  others => <>));
   function Str  (S : String)  return Value_Ref is
      (new Value'(Kind => J_String, S => new String'(S), others => <>));
   function Num  (N : Float)   return Value_Ref is (new Value'(Kind => J_Float, F => N, others => <>));
   function Int  (N : Integer) return Value_Ref is (new Value'(Kind => J_Int, I => N, others => <>));
   function Bool (B : Boolean) return Value_Ref is (new Value'(Kind => J_Bool, B => B, others => <>));
   function Null_Value return Value_Ref is (new Value'(Kind => J_Null, others => <>));

   procedure Set (Obj : Value_Ref; Key : String; Val : Value_Ref) is
      P : constant Pair_Ref := new Pair'(Key => new String'(Key), Val => Val, Next => null);
   begin
      if Obj = null or else Obj.Kind /= J_Object then
         return;
      end if;
      if Obj.Pairs = null then
         Obj.Pairs := P;
      else
         Obj.Tail_P.Next := P;
      end if;
      Obj.Tail_P := P;
   end Set;

   procedure Append (Arr : Value_Ref; Val : Value_Ref) is
   begin
      if Arr = null or else Arr.Kind /= J_Array or else Val = null then
         return;
      end if;
      Val.Next := null;
      if Arr.Items = null then
         Arr.Items := Val;
      else
         Arr.Tail_V.Next := Val;
      end if;
      Arr.Tail_V := Val;
   end Append;

   ----------------------------------------------------------------
   -- Introspection
   ----------------------------------------------------------------
   function Is_Object (V : Value_Ref) return Boolean is (V /= null and then V.Kind = J_Object);
   function Is_Array  (V : Value_Ref) return Boolean is (V /= null and then V.Kind = J_Array);
   function Exists (V : Value_Ref) return Boolean is (V /= null and then V.Kind /= J_Null);

   function Get (V : Value_Ref; Key : String) return Value_Ref is
      P : Pair_Ref;
   begin
      if V = null or else V.Kind /= J_Object then
         return null;
      end if;
      P := V.Pairs;
      while P /= null loop
         if P.Key /= null and then P.Key.all = Key then
            return P.Val;
         end if;
         P := P.Next;
      end loop;
      return null;
   end Get;

   function Length (V : Value_Ref) return Natural is
      N : Natural := 0;
      E : Value_Ref;
   begin
      if V = null or else V.Kind /= J_Array then
         return 0;
      end if;
      E := V.Items;
      while E /= null loop
         N := N + 1;
         E := E.Next;
      end loop;
      return N;
   end Length;

   function Item (V : Value_Ref; I : Positive) return Value_Ref is
      N : Natural := 0;
      E : Value_Ref;
   begin
      if V = null or else V.Kind /= J_Array then
         return null;
      end if;
      E := V.Items;
      while E /= null loop
         N := N + 1;
         if N = I then
            return E;
         end if;
         E := E.Next;
      end loop;
      return null;
   end Item;

   function As_String (V : Value_Ref; Default : String := "") return String is
   begin
      if V /= null and then V.Kind = J_String and then V.S /= null then
         return V.S.all;
      end if;
      return Default;
   end As_String;

   function As_Float (V : Value_Ref; Default : Float := 0.0) return Float is
   begin
      if V = null then return Default; end if;
      case V.Kind is
         when J_Float => return V.F;
         when J_Int   => return Float (V.I);
         when others  => return Default;
      end case;
   end As_Float;

   function As_Int (V : Value_Ref; Default : Integer := 0) return Integer is
   begin
      if V = null then return Default; end if;
      case V.Kind is
         when J_Int   => return V.I;
         when J_Float => return Integer (V.F);
         when others  => return Default;
      end case;
   end As_Int;

   function As_Bool (V : Value_Ref; Default : Boolean := False) return Boolean is
   begin
      if V /= null and then V.Kind = J_Bool then
         return V.B;
      end if;
      return Default;
   end As_Bool;

   ----------------------------------------------------------------
   -- Serializer
   ----------------------------------------------------------------
   procedure Emit_String (Out_S : in out Unbounded_String; S : String) is
   begin
      Append (Out_S, '"');
      for C of S loop
         case C is
            when '"'  => Append (Out_S, "\""");
            when '\'  => Append (Out_S, "\\");
            when LF   => Append (Out_S, "\n");
            when CR   => Append (Out_S, "\r");
            when HT   => Append (Out_S, "\t");
            when others =>
               if Character'Pos (C) < 32 then
                  declare
                     H : constant String := "0123456789abcdef";
                     P : constant Natural := Character'Pos (C);
                  begin
                     Append (Out_S, "\u00");
                     Append (Out_S, H (H'First + P / 16));
                     Append (Out_S, H (H'First + P mod 16));
                  end;
               else
                  Append (Out_S, C);
               end if;
         end case;
      end loop;
      Append (Out_S, '"');
   end Emit_String;

   procedure Emit (Out_S : in out Unbounded_String; V : Value_Ref) is
      First : Boolean;
      E : Value_Ref;
      P : Pair_Ref;
   begin
      if V = null then
         Append (Out_S, "null");
         return;
      end if;
      case V.Kind is
         when J_Null   => Append (Out_S, "null");
         when J_Bool   => Append (Out_S, (if V.B then "true" else "false"));
         when J_Int    => Append (Out_S, Ada.Strings.Unbounded.Trim
                            (To_Unbounded_String (Integer'Image (V.I)), Ada.Strings.Left));
         when J_Float  =>
            declare
               Img : constant String := Float'Image (V.F);
            begin
               Append (Out_S, (if Img (Img'First) = ' ' then Img (Img'First + 1 .. Img'Last) else Img));
            end;
         when J_String => Emit_String (Out_S, (if V.S = null then "" else V.S.all));
         when J_Array  =>
            Append (Out_S, '[');
            First := True; E := V.Items;
            while E /= null loop
               if not First then Append (Out_S, ','); end if;
               First := False;
               Emit (Out_S, E);
               E := E.Next;
            end loop;
            Append (Out_S, ']');
         when J_Object =>
            Append (Out_S, '{');
            First := True; P := V.Pairs;
            while P /= null loop
               if not First then Append (Out_S, ','); end if;
               First := False;
               Emit_String (Out_S, (if P.Key = null then "" else P.Key.all));
               Append (Out_S, ':');
               Emit (Out_S, P.Val);
               P := P.Next;
            end loop;
            Append (Out_S, '}');
      end case;
   end Emit;

   function To_String (V : Value_Ref) return String is
      Out_S : Unbounded_String;
   begin
      Emit (Out_S, V);
      return To_String (Out_S);
   end To_String;

   ----------------------------------------------------------------
   -- Parser
   ----------------------------------------------------------------
   function Parse (S : String) return Value_Ref is
      Pos : Natural := S'First;

      procedure Skip_WS is
      begin
         while Pos <= S'Last
           and then (S (Pos) = ' ' or else S (Pos) = LF
                     or else S (Pos) = CR or else S (Pos) = HT)
         loop
            Pos := Pos + 1;
         end loop;
      end Skip_WS;

      function Parse_Value return Value_Ref;

      function Parse_Str return String is
         R : Unbounded_String;
      begin
         Pos := Pos + 1;   -- opening quote
         while Pos <= S'Last and then S (Pos) /= '"' loop
            if S (Pos) = '\' and then Pos < S'Last then
               Pos := Pos + 1;
               case S (Pos) is
                  when '"'  => Append (R, '"');
                  when '\'  => Append (R, '\');
                  when '/'  => Append (R, '/');
                  when 'n'  => Append (R, LF);
                  when 'r'  => Append (R, CR);
                  when 't'  => Append (R, HT);
                  when 'b'  => Append (R, Character'Val (8));
                  when 'f'  => Append (R, Character'Val (12));
                  when 'u'  =>
                     if Pos + 4 <= S'Last then
                        declare
                           function HexV (C : Character) return Natural is
                             (case C is when '0' .. '9' => Character'Pos (C) - Character'Pos ('0'),
                                        when 'a' .. 'f' => Character'Pos (C) - Character'Pos ('a') + 10,
                                        when 'A' .. 'F' => Character'Pos (C) - Character'Pos ('A') + 10,
                                        when others => 0);
                           CP : constant Natural :=
                             HexV (S (Pos + 1)) * 4096 + HexV (S (Pos + 2)) * 256
                             + HexV (S (Pos + 3)) * 16 + HexV (S (Pos + 4));
                        begin
                           Pos := Pos + 4;
                           if CP < 16#80# then
                              Append (R, Character'Val (CP));
                           elsif CP < 16#800# then
                              Append (R, Character'Val (16#C0# + CP / 64));
                              Append (R, Character'Val (16#80# + CP mod 64));
                           else
                              Append (R, Character'Val (16#E0# + CP / 4096));
                              Append (R, Character'Val (16#80# + (CP / 64) mod 64));
                              Append (R, Character'Val (16#80# + CP mod 64));
                           end if;
                        end;
                     end if;
                  when others => Append (R, S (Pos));
               end case;
            else
               Append (R, S (Pos));
            end if;
            Pos := Pos + 1;
         end loop;
         if Pos > S'Last then raise Parse_Error; end if;
         Pos := Pos + 1;   -- closing quote
         return To_String (R);
      end Parse_Str;

      function Parse_Value return Value_Ref is
      begin
         Skip_WS;
         if Pos > S'Last then raise Parse_Error; end if;
         case S (Pos) is
            when '{' =>
               declare
                  O : constant Value_Ref := New_Object;
               begin
                  Pos := Pos + 1; Skip_WS;
                  if Pos <= S'Last and then S (Pos) = '}' then Pos := Pos + 1; return O; end if;
                  loop
                     Skip_WS;
                     if Pos > S'Last or else S (Pos) /= '"' then raise Parse_Error; end if;
                     declare K : constant String := Parse_Str; begin
                        Skip_WS;
                        if Pos > S'Last or else S (Pos) /= ':' then raise Parse_Error; end if;
                        Pos := Pos + 1;
                        Set (O, K, Parse_Value);
                     end;
                     Skip_WS;
                     exit when Pos > S'Last or else S (Pos) = '}';
                     if S (Pos) /= ',' then raise Parse_Error; end if;
                     Pos := Pos + 1;
                  end loop;
                  if Pos > S'Last then raise Parse_Error; end if;
                  Pos := Pos + 1;   -- '}'
                  return O;
               end;
            when '[' =>
               declare
                  A : constant Value_Ref := New_Array;
               begin
                  Pos := Pos + 1; Skip_WS;
                  if Pos <= S'Last and then S (Pos) = ']' then Pos := Pos + 1; return A; end if;
                  loop
                     Append (A, Parse_Value);
                     Skip_WS;
                     exit when Pos > S'Last or else S (Pos) = ']';
                     if S (Pos) /= ',' then raise Parse_Error; end if;
                     Pos := Pos + 1;
                  end loop;
                  if Pos > S'Last then raise Parse_Error; end if;
                  Pos := Pos + 1;   -- ']'
                  return A;
               end;
            when '"' =>
               return Str (Parse_Str);
            when 't' =>
               Pos := Pos + 4; return Bool (True);
            when 'f' =>
               Pos := Pos + 5; return Bool (False);
            when 'n' =>
               Pos := Pos + 4; return Null_Value;
            when others =>
               declare
                  Start  : constant Natural := Pos;
                  Is_Flt : Boolean := False;
               begin
                  while Pos <= S'Last
                    and then (S (Pos) in '0' .. '9' or else S (Pos) = '-'
                              or else S (Pos) = '+' or else S (Pos) = '.'
                              or else S (Pos) = 'e' or else S (Pos) = 'E')
                  loop
                     if S (Pos) = '.' or else S (Pos) = 'e' or else S (Pos) = 'E' then
                        Is_Flt := True;
                     end if;
                     Pos := Pos + 1;
                  end loop;
                  declare
                     Tok : constant String := S (Start .. Pos - 1);
                  begin
                     if Is_Flt then
                        return Num (Float'Value (Tok));
                     else
                        return Int (Integer'Value (Tok));
                     end if;
                  exception
                     when others => raise Parse_Error;
                  end;
               end;
         end case;
      end Parse_Value;

      Result : constant Value_Ref := Parse_Value;
   begin
      return Result;
   end Parse;

end JSON;
