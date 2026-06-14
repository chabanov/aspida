---------------------------------------------------------------------
-- LLM_Parallel body (generic procedure)
---------------------------------------------------------------------

with System.Multiprocessors;

procedure LLM_Parallel
  (First, Last : Integer;
   Min_Grain   : Integer := 256)
is
   Count : constant Integer := Last - First + 1;
   N_CPU : constant Integer := Integer (System.Multiprocessors.Number_Of_CPUs);
begin
   if Count <= 0 then
      return;
   elsif Count < Min_Grain or else N_CPU <= 1 then
      Work (First, Last);
      return;
   end if;

   declare
      N     : constant Integer := Integer'Min (N_CPU, Count);
      Chunk : constant Integer := (Count + N - 1) / N;

      task type Worker is
         entry Go (Lo, Hi : Integer);
      end Worker;

      task body Worker is
         L, H : Integer;
      begin
         accept Go (Lo, Hi : Integer) do
            L := Lo;  H := Hi;
         end Go;
         Work (L, H);
      end Worker;

      Pool : array (1 .. N) of Worker;
   begin
      for I in 1 .. N loop
         declare
            Lo : constant Integer := First + (I - 1) * Chunk;
            Hi : constant Integer := Integer'Min (First + I * Chunk - 1, Last);
         begin
            if Lo <= Last then
               Pool (I).Go (Lo, Hi);
            else
               Pool (I).Go (1, 0);
            end if;
         end;
      end loop;
   end;   -- waits for all workers
end LLM_Parallel;
