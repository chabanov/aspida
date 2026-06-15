---------------------------------------------------------------------
-- LLM_Sampler body.
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with Ada.Unchecked_Deallocation;

package body LLM_Sampler is

   use type Interfaces.Unsigned_64;
   function Numel (T : LLM_Tensor.Tensor) return Integer renames LLM_Tensor.Numel;
   function Get_Flat (T : LLM_Tensor.Tensor; I : Integer) return Float
     renames LLM_Tensor.Get_Flat;

   function Create (P : Params) return Sampler is
      Seed : constant Interfaces.Unsigned_64 :=
        (if P.Seed = 0 then 16#9E3779B97F4A7C15#
         else Interfaces.Unsigned_64'Mod (P.Seed));
   begin
      --  Avoid the all-zero state (xorshift fixed point).
      return (P => P, State => (if Seed = 0 then 1 else Seed));
   end Create;

   --  xorshift64* — deterministic, self-contained.
   function Next_U64 (S : in out Sampler) return Interfaces.Unsigned_64 is
      use Interfaces;
      X : Unsigned_64 := S.State;
   begin
      X := X xor Shift_Right (X, 12);
      X := X xor Shift_Left  (X, 25);
      X := X xor Shift_Right (X, 27);
      S.State := X;
      return X * 16#2545F4914F6CDD1D#;
   end Next_U64;

   --  Uniform float in [0.0, 1.0).
   function Next_Float (S : in out Sampler) return Float is
      use Interfaces;
      U : constant Unsigned_64 := Shift_Right (Next_U64 (S), 11);  -- 53 bits
   begin
      return Float (U) / Float (2.0 ** 53);
   end Next_Float;

   function Next
     (S      : in out Sampler;
      Logits : LLM_Tensor.Tensor;
      Recent : History := Empty_History) return Integer
   is
      --  Work buffers are HEAP-allocated: the vocab can be 128k–262k, far too
      --  large for a task stack (a server handler task overflowed otherwise).
      type Float_Arr is array (Positive range <>) of Float;
      type Float_Ptr is access Float_Arr;
      procedure Free is new Ada.Unchecked_Deallocation (Float_Arr, Float_Ptr);
      type Bool_Arr is array (Positive range <>) of Boolean;
      type Bool_Ptr is access Bool_Arr;
      procedure Free is new Ada.Unchecked_Deallocation (Bool_Arr, Bool_Ptr);

      N      : constant Integer := Numel (Logits);
      L      : Float_Ptr := new Float_Arr (1 .. N);  -- logits, then probs
      Result : Integer := 0;
   begin
      for I in 1 .. N loop L (I) := Get_Flat (Logits, I); end loop;

      --  Repetition penalty (llama.cpp convention): divide positive logits,
      --  multiply negative ones, for every recently emitted token id.
      if S.P.Repeat_Penalty /= 1.0 then
         for R of Recent loop
            if R >= 0 and then R + 1 <= N then
               if L (R + 1) > 0.0 then
                  L (R + 1) := L (R + 1) / S.P.Repeat_Penalty;
               else
                  L (R + 1) := L (R + 1) * S.P.Repeat_Penalty;
               end if;
            end if;
         end loop;
      end if;

      if S.P.Temperature <= 0.0 then
         --  Greedy: argmax, no RNG draw.
         declare
            Best : Integer := 1;
         begin
            for I in 2 .. N loop
               if L (I) > L (Best) then Best := I; end if;
            end loop;
            Result := Best - 1;
         end;
      else
         --  Temperature scale + numerically stable softmax, in place on L.
         declare
            Mx  : Float := L (1);
            Den : Float := 0.0;
         begin
            for I in 1 .. N loop L (I) := L (I) / S.P.Temperature; end loop;
            for I in 2 .. N loop if L (I) > Mx then Mx := L (I); end if; end loop;
            for I in 1 .. N loop
               L (I) := Exp (L (I) - Mx);
               Den := Den + L (I);
            end loop;
            for I in 1 .. N loop L (I) := L (I) / Den; end loop;
         end;

         if S.P.Top_K <= 0 and then S.P.Top_P >= 1.0 then
            --  Pure temperature: draw straight from the full distribution.
            declare
               U : constant Float := Next_Float (S); Acc : Float := 0.0;
            begin
               Result := N - 1;
               for I in 1 .. N loop
                  Acc := Acc + L (I);
                  if U <= Acc then Result := I - 1; exit; end if;
               end loop;
            end;
         else
            --  Top-EffK partial selection (O(N*EffK)), then top-p, then draw.
            declare
               EffK : constant Integer :=
                 (if S.P.Top_K > 0 then Integer'Min (S.P.Top_K, N)
                  else Integer'Min (N, 256));
               Idx  : array (1 .. EffK) of Integer := [others => 0];
               Used : Bool_Ptr := new Bool_Arr (1 .. N);
               Keep : Integer := EffK;
            begin
               for I in 1 .. N loop Used (I) := False; end loop;
               for R in 1 .. EffK loop
                  declare Best : Integer := 0; begin
                     for I in 1 .. N loop
                        if not Used (I)
                          and then (Best = 0 or else L (I) > L (Best))
                        then
                           Best := I;
                        end if;
                     end loop;
                     Idx (R) := Best; Used (Best) := True;
                  end;
               end loop;

               --  Top-p (nucleus): smallest prefix whose mass >= Top_P.
               if S.P.Top_P < 1.0 then
                  declare Cum : Float := 0.0; begin
                     for R in 1 .. EffK loop
                        Cum := Cum + L (Idx (R));
                        if Cum >= S.P.Top_P then Keep := R; exit; end if;
                     end loop;
                  end;
               end if;

               --  Renormalise the kept set and draw.
               declare
                  KDen : Float := 0.0;
                  U    : constant Float := Next_Float (S);
                  Acc  : Float := 0.0;
               begin
                  for R in 1 .. Keep loop KDen := KDen + L (Idx (R)); end loop;
                  Result := Idx (Keep) - 1;   -- fallback (rounding)
                  for R in 1 .. Keep loop
                     Acc := Acc + L (Idx (R)) / KDen;
                     if U <= Acc then Result := Idx (R) - 1; exit; end if;
                  end loop;
               end;
               Free (Used);
            end;
         end if;
      end if;

      Free (L);
      return Result;
   end Next;

end LLM_Sampler;
