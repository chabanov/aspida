---------------------------------------------------------------------
-- Test RMSNorm validity
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_RMSNorm;
with LLM_Tensor; use LLM_Tensor;

procedure Test_RMSNorm is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Assert;

   function Float_Close (A, B : Float; Tol : Float := 1.0e-4) return Boolean is
   begin
      return abs (A - B) < Tol;
   end Float_Close;

begin
   Put_Line ("=== RMSNorm Test Suite ===");
   New_Line;

   -- Test 1: zero input, weight=1 → all zeros
   declare
      X : Tensor := New_Tensor ([1, 4]);
      W : Tensor := New_Tensor ([1, 4]);
      Result : Tensor;
   begin
      for I in 1 .. 4 loop
         Set_Flat (X, I, 0.0);
         Set_Flat (W, I, 1.0);
      end loop;
      Result := LLM_RMSNorm.Forward (X, W);
      for I in 1 .. 4 loop
         Assert ("Zero input [" & Integer'Image (I) & "]",
           Float_Close (Get_Flat (Result, I), 0.0, 1.0e-3));
      end loop;
   end;

   -- Test 2: all-ones input, weight=2 → all ≈ 2.0
   declare
      X : Tensor := New_Tensor ([1, 4]);
      W : Tensor := New_Tensor ([1, 4]);
      Y : Tensor;
   begin
      for I in 1 .. 4 loop
         Set_Flat (X, I, 1.0);
         Set_Flat (W, I, 2.0);
      end loop;
      Y := LLM_RMSNorm.Forward (X, W);
      for I in 1 .. 4 loop
         Assert ("All-ones × 2.0 [" & Integer'Image (I) & "]",
           Float_Close (Get_Flat (Y, I), 2.0, 1.0e-3));
      end loop;
   end;

   -- Test 3: variable input, weight=1
   -- x = [1.0, 2.0, 3.0, 4.0]
   -- mean(x²) = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5+eps) ≈ 2.739
   -- out ≈ [0.365, 0.730, 1.095, 1.460]
   declare
      X : Tensor := New_Tensor ([1, 4]);
      W : Tensor := New_Tensor ([1, 4]);
      Expected : constant array (1 .. 4) of Float := [0.365, 0.730, 1.095, 1.460];
      Y : Tensor;
   begin
      Set_Flat (X, 1, 1.0); Set_Flat (X, 2, 2.0);
      Set_Flat (X, 3, 3.0); Set_Flat (X, 4, 4.0);
      for I in 1 .. 4 loop
         Set_Flat (W, I, 1.0);
      end loop;
      Y := LLM_RMSNorm.Forward (X, W);
      for I in 1 .. 4 loop
         Assert ("Scaled test [" & Integer'Image (I) & "]",
           Float_Close (Get_Flat (Y, I), Expected (I), 1.0e-2));
      end loop;
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_RMSNorm;
