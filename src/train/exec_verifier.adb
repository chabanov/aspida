------------------------------------------------------------------------
-- Exec_Verifier body — shells out to python3 via /bin/sh (child stdout/stderr
-- suppressed) and treats exit code 0 (all asserts held) as "correct".
------------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Directories;
with Ada.Environment_Variables;
with GNAT.OS_Lib;

package body Exec_Verifier is

   LF : constant Character := ASCII.LF;
   Py : constant GNAT.OS_Lib.String_Access :=
     GNAT.OS_Lib.Locate_Exec_On_Path ("python3");
   --  Wall-clock bound for candidate execution (rejects hangs). Null on hosts
   --  without coreutils `timeout` (e.g. macOS) — then we run without a bound.
   TO : constant GNAT.OS_Lib.String_Access :=
     GNAT.OS_Lib.Locate_Exec_On_Path ("timeout");
   Run_Seq : Natural := 0;   -- unique scratch path per execution

   function Available return Boolean is (GNAT.OS_Lib."/=" (Py, null));

   function Name (P : Problem_Id) return String is
     (case P is
        when 1 => "add(a,b) = a+b",
        when 2 => "is_even(n)",
        when 3 => "max_of(lst)",
        when 4 => "reverse_str(s)",
        when 5 => "factorial(n)");

   function Prompt (P : Problem_Id) return String is
     (case P is
        when 1 => "Write a Python function add(a, b) that returns the sum of a and b.",
        when 2 => "Write a Python function is_even(n) that returns True if n is even, otherwise False.",
        when 3 => "Write a Python function max_of(lst) that returns the largest element of the list lst.",
        when 4 => "Write a Python function reverse_str(s) that returns the string s reversed.",
        when 5 => "Write a Python function factorial(n) that returns n factorial, with factorial(0) == 1.");

   --  Per-problem test harness (Python asserts that raise on failure).
   function Tests (P : Problem_Id) return String is
     (case P is
        when 1 =>
          "assert add(2,3)==5" & LF & "assert add(-1,1)==0" & LF,
        when 2 =>
          "assert is_even(4)" & LF & "assert not is_even(3)" & LF,
        when 3 =>
          "assert max_of([1,5,2])==5" & LF & "assert max_of([-3,-1])==-1" & LF,
        when 4 =>
          "assert reverse_str('abc')=='cba'" & LF
          & "assert reverse_str('')==''" & LF,
        when 5 =>
          "assert factorial(5)==120" & LF & "assert factorial(0)==1" & LF);

   --  HELD-OUT tests: DIFFERENT inputs from the visible set above, used only
   --  for evaluation (never shown for selection) — defeats overfitting.
   function Hidden_Tests (P : Problem_Id) return String is
     (case P is
        when 1 =>
          "assert add(10,20)==30" & LF & "assert add(0,0)==0" & LF
          & "assert add(123,-23)==100" & LF,
        when 2 =>
          "assert is_even(8)" & LF & "assert not is_even(7)" & LF
          & "assert is_even(0)" & LF,
        when 3 =>
          "assert max_of([4,9,2,7])==9" & LF & "assert max_of([5])==5" & LF
          & "assert max_of([-9,-2,-5])==-2" & LF,
        when 4 =>
          "assert reverse_str('hello')=='olleh'" & LF
          & "assert reverse_str('xy')=='yx'" & LF,
        when 5 =>
          "assert factorial(4)==24" & LF & "assert factorial(1)==1" & LF
          & "assert factorial(6)==720" & LF);

   --  Write Code to a UNIQUE scratch file and run it directly (no shell-string
   --  interpolation), child output suppressed, wall-clock bounded by `timeout`
   --  when available; True iff it exits 0.
   --
   --  Split a space-separated command prefix into argv tokens (heap String_Access).
   function Split (S : String) return GNAT.OS_Lib.Argument_List is
      N : Natural := 0;
      I : Integer := S'First;
   begin
      while I <= S'Last loop
         if S (I) = ' ' then I := I + 1;
         else N := N + 1; while I <= S'Last and then S (I) /= ' ' loop I := I + 1; end loop;
         end if;
      end loop;
      declare
         R : GNAT.OS_Lib.Argument_List (1 .. N);
         K : Natural := 0;
         J : Integer := S'First;
      begin
         while J <= S'Last loop
            if S (J) = ' ' then J := J + 1;
            else
               declare St : constant Integer := J;
               begin
                  while J <= S'Last and then S (J) /= ' ' loop J := J + 1; end loop;
                  K := K + 1; R (K) := new String'(S (St .. J - 1));
               end;
            end if;
         end loop;
         return R;
      end;
   end Split;

   --  NOTE: this runs MODEL-GENERATED code. The in-process hardening (unique
   --  path, direct exec, no shell, timeout) is a down-payment. For MULTI-TENANT
   --  / rented-droplet use set ASPIDA_VERIFY_SANDBOX to an isolator command
   --  prefix (microVM/container/firejail: non-root, read-only rootfs, no network
   --  + blocked cloud-metadata, rlimits/cgroups, seccomp) — every execution is
   --  then wrapped by it. See PLATFORM.md §Security for the required recipe.
   function Run (Code : String) return Boolean is
      use GNAT.OS_Lib;
      Path : constant String := "/tmp/aspida_verify_" &
        Ada.Strings.Fixed.Trim (Natural'Image (Run_Seq), Ada.Strings.Left) & ".py";
      F  : Ada.Text_IO.File_Type;
      Ok : Boolean;
      Rc : Integer;
      Sb : constant String :=
        Ada.Environment_Variables.Value ("ASPIDA_VERIFY_SANDBOX", "");
      --  full argv = [sandbox…] [timeout 10] python3 <path>
      Sand : constant Argument_List := Split (Sb);
      Tmo  : constant Argument_List :=
        (if GNAT.OS_Lib."/=" (TO, null)
         then [new String'(TO.all), new String'("10")]
         else [1 .. 0 => <>]);
      Exe  : constant Argument_List := [new String'(Py.all), new String'(Path)];
      Full : Argument_List := Sand & Tmo & Exe;   -- mutable so elements can be Free'd
   begin
      if not Available then
         for A of Full loop Free (A); end loop;
         return False;
      end if;
      Run_Seq := Run_Seq + 1;
      Ada.Text_IO.Create (F, Ada.Text_IO.Out_File, Path);
      Ada.Text_IO.Put (F, Code);
      Ada.Text_IO.Close (F);

      Spawn (Full (Full'First).all, Full (Full'First + 1 .. Full'Last),
             "/dev/null", Ok, Rc, Err_To_Out => True);
      for A of Full loop Free (A); end loop;

      begin
         Ada.Directories.Delete_File (Path);   -- best-effort scratch cleanup
      exception
         when others => null;
      end;
      return Rc = 0;
   end Run;

   overriding function Is_Correct
     (V : Python_Verifier; Spec : Natural; Source : String) return Boolean
   is
      pragma Unreferenced (V);
   begin
      if Spec not in Problem_Id then
         return False;
      end if;
      return Run (Source & LF & Tests (Spec));
   end Is_Correct;

   function Eval_Correct (Spec : Problem_Id; Source : String) return Boolean is
   begin
      return Run (Source & LF & Hidden_Tests (Spec));
   end Eval_Correct;

end Exec_Verifier;
