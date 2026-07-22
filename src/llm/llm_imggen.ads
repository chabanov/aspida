--  LLM_ImgGen — image gen/edit support helpers for the aspida engine.
--
--  Image generation itself runs OUT OF PROCESS in aspida-imgd (see Img_Daemon)
--  so stable-diffusion.cpp's CUDA backend can never interpose on the LLM's.
--  This package holds only the pure-Ada pieces secure_server needs on either
--  side of that call: a cheap availability probe and the base64 codecs for the
--  reference (in) and result (out) images. It links NO native image library.
package LLM_ImgGen is

   --  Whether the image model files are present on disk. Cheap; loads nothing.
   --  Used to answer "image model not installed" without dialing the daemon.
   function Available return Boolean;

   --  Base64 (standard alphabet) of a file's bytes — for the PNG response.
   function Encode_File_B64 (Path : String) return String;

   --  Decode a base64 (or "data:...;base64,XXXX") string to a temp PNG file;
   --  returns its path ("" on failure). For the edit reference image.
   function Decode_B64_To_Temp (B64 : String) return String;

end LLM_ImgGen;
