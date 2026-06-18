---------------------------------------------------------------------
-- Export_Llama — write a single-block, single-head student (Train weights) as
-- a llama-architecture GGUF the Aspida engine can load. Weights use the
-- training layout [in, out]; this transposes them to GGUF [ne0=in, ne1=out]
-- (Rows=out) and emits the required hyperparam metadata + tokenizer.
---------------------------------------------------------------------

with Train;
with GGUF_Write;

package Export_Llama is

   --  Config (Dim, FFN, Vocab) is derived from the weight shapes:
   --    E[Vocab,Dim]  Wg[Dim,FFN]  ...  norms are [1,Dim].
   procedure Save
     (Path : String;
      E, G1, G2, Gf, Wq, Wk, Wv, Wo, Wg, Wu, Wd, Wout : Train.Matrix;
      Tokens    : GGUF_Write.Str_List;
      Bos, Eos  : Natural := 1;
      Ctx       : Natural := 64;
      Rope_Base : Float   := 10000.0;
      RMS_Eps   : Float   := 1.0E-5);

end Export_Llama;
