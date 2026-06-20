---------------------------------------------------------------------
-- Ctx_Window body.
---------------------------------------------------------------------

package body Ctx_Window is

   procedure Select_Messages
     (Lengths      : Len_Array;
      System_First : Boolean;
      Overhead     : Natural;
      Budget       : Natural;
      Keep         : out Keep_Array;
      Overflow     : out Boolean)
   is
      Lo   : constant Positive := Lengths'First;
      Hi   : constant Positive := Lengths'Last;
      Used : Natural := Overhead;

      function Take (I : Positive) return Boolean is
         --  Include message I if it still fits; record it.
      begin
         if Keep (I) then return True; end if;       -- already counted
         if Used + Lengths (I) <= Budget then
            Used := Used + Lengths (I);
            Keep (I) := True;
            return True;
         end if;
         return False;
      end Take;

      Forced_OK : Boolean := True;
   begin
      Keep := [others => False];
      Overflow := False;

      --  Pin the system prompt (if any) and the newest message unconditionally.
      if System_First then
         Keep (Lo) := True;
         Used := Used + Lengths (Lo);
      end if;
      if not Keep (Hi) then
         Keep (Hi) := True;
         Used := Used + Lengths (Hi);
      end if;

      --  If the pinned essentials already exceed the budget, report overflow
      --  (the caller decides whether to error or send the best-effort prompt).
      if Used > Budget then
         Forced_OK := False;
      end if;

      --  Fill the rest with the most recent non-pinned turns (newest first), so
      --  the freshest context is preserved and the oldest is dropped. When
      --  there is no system prompt the oldest message (Lo) is a normal turn and
      --  must remain eligible, so the range starts at Lo in that case.
      --
      --  Best-fit: we keep walking older turns after a miss rather than
      --  stopping, so a small older turn can still be reclaimed when a larger
      --  recent one did not fit. Newer turns are still preferred (reverse
      --  iteration order), so this only ever adds context that the old
      --  stop-on-first-miss loop would have dropped.
      if Forced_OK then
         declare
            Mid_Lo : constant Positive := (if System_First then Lo + 1 else Lo);
            Dummy  : Boolean;
            pragma Unreferenced (Dummy);
         begin
            for I in reverse Mid_Lo .. Hi - 1 loop
               Dummy := Take (I);
            end loop;
         end;
      end if;

      Overflow := not Forced_OK;
   end Select_Messages;

end Ctx_Window;
