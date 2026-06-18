---------------------------------------------------------------------
-- Aspida — Type-safe HTTP client generator
-- Root package specification with SPARK annotations
---------------------------------------------------------------------

with Ada.Strings.Unbounded;

package Aspida with
  SPARK_Mode => On,
  Abstract_State => (Config),
  Initializes    => Config
is

   -------------------------------------------------------------------
   -- Types
   -------------------------------------------------------------------

   -- HTTP methods supported by the generator
   type Http_Method is (GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD);

   -- URL segments and query construction
   subtype Segment is String;

   -- Parameter source: where the value ends up in the HTTP request
   type Param_Kind is (Path_Param, Query_Param, Header_Param, Cookie_Param);

   -- Type-safe parameter descriptor
   type Param_Descriptor is record
      Name : Ada.Strings.Unbounded.Unbounded_String;
      Kind : Param_Kind;
      Type_Name : Ada.Strings.Unbounded.Unbounded_String;
      Optional : Boolean := False;
   end record;

   -- A single API endpoint definition
   type Endpoint is record
      Method   : Http_Method;
      Path     : Ada.Strings.Unbounded.Unbounded_String;
      Summary  : Ada.Strings.Unbounded.Unbounded_String;
      Params   : Param_Descriptor;
      Req_Body_Type : Ada.Strings.Unbounded.Unbounded_String;
      Res_Body_Type : Ada.Strings.Unbounded.Unbounded_String;
   end record;

   -- The parsed API specification
   type Api_Spec is array (Positive range <>) of Endpoint;

   -------------------------------------------------------------------
   -- Core Operations
   -------------------------------------------------------------------

   -- Initialize the generator configuration from a specification file
   procedure Initialize (Spec_Path : String) with
     Global => (Output => Config),
     Pre    => Spec_Path'Length > 0;

   -- Generate TypeScript client code from the current spec
   procedure Generate with
     Global => (Input => Config);

   -- Validate the API specification for consistency
   function Is_Valid (Spec : Api_Spec) return Boolean with
     Global => null;

   -- Parse an OpenAPI v3 document into an Api_Spec
   function Parse_OpenAPI (Path : String) return Api_Spec with
     Global => null,
     Pre    => Path'Length > 0;

   -- Parse a custom aspida TypeScript definition (like the original TS project)
   function Parse_Aspida_TS (Path : String) return Api_Spec with
     Global => null,
     Pre    => Path'Length > 0;

end Aspida;
