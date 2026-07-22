------------------------------------------------------------------------------
--  Img_Daemon — client to the isolated image-generation process (aspida-imgd).
--
--  Image generation runs in a SEPARATE process with its OWN CUDA context, so
--  stable-diffusion.cpp's ggml-CUDA backend can never interpose on the LLM's
--  ggml backend inside secure_server (that in-process coexistence caused
--  illegal-memory-access crashes / wedges). secure_server forwards the request
--  here over a localhost socket and reads back only a success flag; the daemon
--  writes the PNG to Out_Path, which secure_server then base64-encodes.
--
--  Generate never raises: a dead, slow, or hung daemon yields False (a receive
--  timeout guarantees secure_server cannot block indefinitely on image work).
------------------------------------------------------------------------------

package Img_Daemon is

   function Generate
     (Prompt   : String;
      Ref_Path : String := "";      --  empty => text-to-image
      Width    : Integer := 1024;
      Height   : Integer := 1024;
      Steps    : Integer := 20;
      Seed     : Long_Long_Integer := -1;
      Out_Path : String)
      return Boolean;

end Img_Daemon;
