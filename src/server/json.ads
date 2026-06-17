---------------------------------------------------------------------
-- JSON — a small, self-contained JSON parser + builder (no third-party libs).
--
-- Enough for the OpenAI-compatible API: parse a request object and build a
-- response object. Values are reference handles (Value_Ref); the tree is heap
-- allocated and lives for the request (not freed — request-scoped).
---------------------------------------------------------------------

package JSON is

   type Value is private;
   type Value_Ref is access all Value;

   Parse_Error : exception;

   --  Parse a complete JSON document. Raises Parse_Error on malformed input.
   function Parse (S : String) return Value_Ref;

   --  Introspection (all tolerant: wrong-kind / missing returns the Default or
   --  a null Value_Ref, never raises).
   function Is_Object (V : Value_Ref) return Boolean;
   function Is_Array  (V : Value_Ref) return Boolean;
   function Get    (V : Value_Ref; Key : String) return Value_Ref;  -- null if absent
   function Length (V : Value_Ref) return Natural;                  -- array length
   function Item   (V : Value_Ref; I : Positive) return Value_Ref;  -- array element
   function As_String (V : Value_Ref; Default : String := "") return String;
   function As_Float  (V : Value_Ref; Default : Float := 0.0) return Float;
   function As_Int    (V : Value_Ref; Default : Integer := 0) return Integer;
   function As_Bool   (V : Value_Ref; Default : Boolean := False) return Boolean;
   function Exists    (V : Value_Ref) return Boolean;  -- non-null and not JSON null

   --  Building.
   function New_Object return Value_Ref;
   function New_Array  return Value_Ref;
   function Str  (S : String)  return Value_Ref;
   function Num  (N : Float)   return Value_Ref;
   function Int  (N : Integer) return Value_Ref;
   function Bool (B : Boolean) return Value_Ref;
   function Null_Value return Value_Ref;
   procedure Set    (Obj : Value_Ref; Key : String; Val : Value_Ref);
   procedure Append (Arr : Value_Ref; Val : Value_Ref);

   --  Serialize to compact JSON text.
   function To_String (V : Value_Ref) return String;

private

   type Kind_Type is (J_Null, J_Bool, J_Int, J_Float, J_String, J_Array, J_Object);

   type Str_Ptr is access String;

   type Pair;
   type Pair_Ref is access all Pair;

   type Value is record
      Kind   : Kind_Type := J_Null;
      B      : Boolean := False;
      I      : Integer := 0;
      F      : Float := 0.0;
      S      : Str_Ptr := null;
      --  Array: singly-linked list of elements; Object: list of pairs.
      Items  : Value_Ref := null;   -- array head (via Next)
      Pairs  : Pair_Ref := null;    -- object head
      Next   : Value_Ref := null;   -- next array element
      Tail_V : Value_Ref := null;   -- array tail (for O(1) append)
      Tail_P : Pair_Ref := null;    -- object tail
   end record;

   type Pair is record
      Key  : Str_Ptr := null;
      Val  : Value_Ref := null;
      Next : Pair_Ref := null;
   end record;

end JSON;
