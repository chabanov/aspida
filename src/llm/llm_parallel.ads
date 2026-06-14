---------------------------------------------------------------------
-- LLM_Parallel — minimal data-parallel range split over CPUs (Ada tasks).
-- Self-contained: no external thread libraries.
--
-- Work is called with disjoint sub-ranges [Lo, Hi] of [First, Last], so it
-- may write its slice without locking. Runs serially when the range is
-- smaller than Min_Grain (tiny loops avoid task overhead).
---------------------------------------------------------------------

generic
   with procedure Work (Lo, Hi : Integer);
procedure LLM_Parallel
  (First, Last : Integer;
   Min_Grain   : Integer := 256);
