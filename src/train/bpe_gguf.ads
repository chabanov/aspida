---------------------------------------------------------------------
-- BPE_GGUF — embed a learned BPE vocabulary into a GGUF as a byte-level
-- (GPT-2) tokenizer, in exactly the representation LLM_Tokenizer loads:
-- tokenizer.ggml.model = "gpt2", a token list and a merge list mapped through
-- the byte<->unicode bijection. This is the bridge that lets a model trained
-- from scratch (with its own learned vocabulary) be served by our own engine.
---------------------------------------------------------------------

with GGUF_Write;
with BPE_Train;

package BPE_GGUF is

   --  Add the tokenizer metadata for trainer T to GGUF builder B. Call before
   --  Save, alongside the model tensors / metadata.
   procedure Write_Tokenizer
     (B : in out GGUF_Write.Builder; T : BPE_Train.Trainer);

end BPE_GGUF;
