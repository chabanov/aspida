---------------------------------------------------------------------
-- Aspida — Package body (stub)
---------------------------------------------------------------------

package body Aspida with
  SPARK_Mode => On,
  Refined_State => (Config => Config_Initialized)
is

   -------------------------------------------------------------------
   -- Declarations
   -------------------------------------------------------------------

   Config_Initialized : Boolean := False;

   -------------------------------------------------------------------
   -- Initialize
   -------------------------------------------------------------------

   procedure Initialize (Spec_Path : String) is
      pragma Unreferenced (Spec_Path);
   begin
      Config_Initialized := True;
   end Initialize;

   -------------------------------------------------------------------
   -- Generate
   -------------------------------------------------------------------

   procedure Generate is
   begin
      if not Config_Initialized then
         raise Program_Error with "Aspida not initialized — call Initialize first";
      end if;

      -- TODO: Walk the parsed spec and emit TypeScript client code
      null;
   end Generate;

   -------------------------------------------------------------------
   -- Is_Valid
   -------------------------------------------------------------------

   function Is_Valid (Spec : Api_Spec) return Boolean is
   begin
      -- TODO: Validate endpoint paths, param consistency, etc.
      return Spec'Length > 0;
   end Is_Valid;

   -------------------------------------------------------------------
   -- Parse_OpenAPI
   -------------------------------------------------------------------

   function Parse_OpenAPI (Path : String) return Api_Spec is
      pragma Unreferenced (Path);
      Result : Api_Spec (1 .. 0);
   begin
      return Result;
   end Parse_OpenAPI;

   -------------------------------------------------------------------
   -- Parse_Aspida_TS
   -------------------------------------------------------------------

   function Parse_Aspida_TS (Path : String) return Api_Spec is
      pragma Unreferenced (Path);
      Result : Api_Spec (1 .. 0);
   begin
      return Result;
   end Parse_Aspida_TS;

end Aspida;
