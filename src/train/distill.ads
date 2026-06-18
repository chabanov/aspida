---------------------------------------------------------------------
-- Distill — the teacher->student distillation dataset.
--
-- "New models are taught by existing models": an existing model (the teacher,
-- run through the Aspida inference engine) produces, for every position of a
-- token sequence, its predicted next-token distribution. Storing the full
-- vocabulary (128k-256k) per position is wasteful, so we keep only the top-K
-- (id, logit) pairs — enough for the student to match the teacher's shape via
-- a sparse KL loss (Phase 3).
--
-- This package is decoupled from the heavy LLM packages: a teacher is any
-- implementor of the Teacher interface that can return full-vocabulary logits
-- for a sequence. The real adapter (a logits-exposing forward over a GGUF
-- backend) plugs in here; tests use a synthetic teacher. The student and the
-- teacher MUST share the same vocabulary/tokenizer for the logits to align.
---------------------------------------------------------------------

with Ada.Containers.Indefinite_Vectors;

package Distill is

   type Token is new Natural;                 -- 0-based token id
   type Token_Array is array (Positive range <>) of Token;

   type Logit        is new Float;            -- 32-bit teacher logit
   type Id_Matrix    is array (Positive range <>, Positive range <>) of Token;
   type Logit_Matrix is array (Positive range <>, Positive range <>) of Logit;
   type Prob_Vector  is array (Positive range <>) of Long_Float;

   --  One training sequence + the teacher's top-K next-token distribution at
   --  each of its N positions.
   type Sample (N, K : Positive) is record
      Tokens    : Token_Array  (1 .. N);              -- input token sequence
      Top_Ids   : Id_Matrix    (1 .. N, 1 .. K);      -- teacher's K argmax ids
      Top_Logit : Logit_Matrix (1 .. N, 1 .. K);      -- their logits
   end record;

   package Sample_Vectors is
     new Ada.Containers.Indefinite_Vectors (Positive, Sample);
   subtype Dataset is Sample_Vectors.Vector;

   --------------------------------------------------------------------
   --  Teacher: returns full-vocabulary logits [N x Vocab] for a sequence.
   --  The real inference engine implements this (see the adapter notes in
   --  the body); tests use a synthetic source.
   --------------------------------------------------------------------
   type Teacher is limited interface;
   function  Vocab   (T : Teacher) return Positive is abstract;
   procedure Forward (T : in out Teacher;
                      Tokens     : Token_Array;
                      Out_Logits : out Logit_Matrix) is abstract;

   --  Run the teacher over Tokens, keep the top-K per position -> one Sample.
   function Capture
     (T : in out Teacher'Class; Tokens : Token_Array; K : Positive)
      return Sample
     with Pre => K <= Vocab (T) and then Tokens'Length >= 1;

   --  The teacher distribution at a position, reconstructed from the stored
   --  top-K logits: softmax over the K kept logits at temperature Temp. The
   --  student gathers its own logits at Top_Ids and matches this (sparse KL).
   function Teacher_Prob
     (S : Sample; Row : Positive; Temp : Long_Float := 1.0) return Prob_Vector
     with Pre => Row in 1 .. S.N;

   --------------------------------------------------------------------
   --  On-disk dataset (local training cache). Binary; round-trips on the
   --  host. (At-rest sealing can wrap this with the existing At_Rest layer.)
   --------------------------------------------------------------------
   procedure Write (Path : String; Data : Dataset);
   function  Read  (Path : String) return Dataset;

end Distill;
