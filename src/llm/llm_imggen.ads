--  LLM_ImgGen — native image generation/editing for the aspida engine.
--  Thin Ada FFI over libaspida_imggen.so (stable-diffusion.cpp / Qwen-Image-
--  Edit-2511 on its OWN isolated ggml). The .so exports only aspida_img_*;
--  its ggml symbols are hidden (version script) so they cannot interpose with
--  the engine's own ggml. Model loads lazily on the first request (~1 min).
package LLM_ImgGen is

   --  Whether the native image library is available (models present + .so
   --  loadable). Cheap; does not load the model.
   function Available return Boolean;

   --  Generate (Ref_Path = "") or edit (Ref_Path /= "") an image; writes a PNG
   --  to Out_Path. Returns True on success. Loads the model on the first call
   --  (serialised, idempotent). Thread-safe.
   function Generate
     (Prompt   : String;
      Ref_Path : String := "";        --  "" => text-to-image
      Width    : Integer := 1024;
      Height   : Integer := 1024;
      Steps    : Integer := 20;
      Cfg      : Float   := 2.5;
      Seed     : Long_Long_Integer := -1;
      Out_Path : String)
      return Boolean;

   --  Base64 (standard alphabet) of a file's bytes — for the PNG response.
   function Encode_File_B64 (Path : String) return String;

   --  Decode a base64 (or "data:...;base64,XXXX") string to a temp PNG file;
   --  returns its path ("" on failure). For the edit reference image.
   function Decode_B64_To_Temp (B64 : String) return String;

end LLM_ImgGen;
