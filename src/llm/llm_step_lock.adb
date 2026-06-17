---------------------------------------------------------------------
-- LLM_Step_Lock body — a simple binary semaphore (mutex) as a protected
-- object. The entry queue is FIFO, so steps from competing generations
-- are served in turn → fair token-by-token interleaving.
---------------------------------------------------------------------

package body LLM_Step_Lock is

   protected Gate is
      entry Acquire;
      procedure Release;
   private
      Held : Boolean := False;
   end Gate;

   protected body Gate is
      entry Acquire when not Held is
      begin
         Held := True;
      end Acquire;

      procedure Release is
      begin
         Held := False;
      end Release;
   end Gate;

   procedure Acquire is
   begin
      Gate.Acquire;
   end Acquire;

   procedure Release is
   begin
      Gate.Release;
   end Release;

end LLM_Step_Lock;
