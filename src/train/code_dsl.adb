------------------------------------------------------------------------
-- Code_DSL body.
------------------------------------------------------------------------

package body Code_DSL is

   A_Tok  : constant := 6;
   B_Tok  : constant := 7;
   Op_Add : constant := 8;
   Op_Sub : constant := 9;
   Op_Mul : constant := 10;
   Op_Min : constant := 11;
   Op_Max : constant := 12;

   function Spec_Token (S : Spec_Id) return Integer is (S);

   function Target (S : Spec_Id; A, B : Integer) return Integer is
   begin
      case S is
         when 1 => return A + B;
         when 2 => return A - B;
         when 3 => return A * B;
         when 4 => return Integer'Min (A, B);
         when 5 => return Integer'Max (A, B);
      end case;
   end Target;

   procedure Run (P : Program; A, B : Integer; Val : out Integer; Ok : out Boolean) is
      function Operand (T : Integer) return Integer is
        (if T = A_Tok then A else B);
      X, Y : Integer;
   begin
      Val := 0;
      Ok  := (P (1) = A_Tok or else P (1) = B_Tok)
        and then (P (2) = A_Tok or else P (2) = B_Tok)
        and then (P (3) in Op_Add .. Op_Max);
      if not Ok then
         return;
      end if;
      X := Operand (P (1));
      Y := Operand (P (2));
      case P (3) is
         when Op_Add => Val := X + Y;
         when Op_Sub => Val := X - Y;
         when Op_Mul => Val := X * Y;
         when Op_Min => Val := Integer'Min (X, Y);
         when Op_Max => Val := Integer'Max (X, Y);
         when others => Ok := False;
      end case;
   end Run;

   function Verify (S : Spec_Id; P : Program) return Boolean is
      Tests : constant array (1 .. 6, 1 .. 2) of Integer :=
        [[3, 5], [7, 2], [4, 4], [9, 1], [2, 8], [6, 3]];
      Val : Integer;
      Ok  : Boolean;
   begin
      for I in Tests'Range (1) loop
         Run (P, Tests (I, 1), Tests (I, 2), Val, Ok);
         if not Ok or else Val /= Target (S, Tests (I, 1), Tests (I, 2)) then
            return False;
         end if;
      end loop;
      return True;
   end Verify;

   function Golden (S : Spec_Id) return Program is
   begin
      case S is
         when 1 => return [A_Tok, B_Tok, Op_Add];   -- a + b
         when 2 => return [A_Tok, B_Tok, Op_Sub];   -- a - b
         when 3 => return [A_Tok, B_Tok, Op_Mul];   -- a * b
         when 4 => return [A_Tok, B_Tok, Op_Min];   -- min(a,b)
         when 5 => return [A_Tok, B_Tok, Op_Max];   -- max(a,b)
      end case;
   end Golden;

   --  A plausible SYSTEMATIC mistake per spec (wrong op or swapped operands).
   function Distractor (S : Spec_Id) return Program is
   begin
      case S is
         when 1 => return [A_Tok, B_Tok, Op_Mul];   -- + mistaken for *
         when 2 => return [B_Tok, A_Tok, Op_Sub];   -- operands swapped (b - a)
         when 3 => return [A_Tok, B_Tok, Op_Add];   -- * mistaken for +
         when 4 => return [A_Tok, B_Tok, Op_Max];   -- min mistaken for max
         when 5 => return [A_Tok, B_Tok, Op_Min];   -- max mistaken for min
      end case;
   end Distractor;

   overriding function Is_Correct
     (V : DSL_Verifier; Spec : Natural; Program : Verifier.Token_Array)
      return Boolean
   is
      pragma Unreferenced (V);
      P : Code_DSL.Program;
   begin
      if Program'Length /= P'Length or else Spec not in Spec_Id then
         return False;
      end if;
      for I in P'Range loop
         P (I) := Program (Program'First + I - 1);
      end loop;
      return Verify (Spec, P);
   end Is_Correct;

end Code_DSL;
