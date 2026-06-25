------------------------------------------------------------------------
-- Platform_Auth body — validate a UARP API key via curl GET /api/v1/me.
------------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Directories;
with Ada.Environment_Variables;
with Interfaces.C;          use Interfaces.C;
with Interfaces.C.Strings;  use Interfaces.C.Strings;
with GNAT.OS_Lib;

package body Platform_Auth is

   Curl : constant GNAT.OS_Lib.String_Access :=
     GNAT.OS_Lib.Locate_Exec_On_Path ("curl");

   function Available return Boolean is (GNAT.OS_Lib."/=" (Curl, null));

   function c_chmod (Path : chars_ptr; Mode : int) return int
     with Import, Convention => C, External_Name => "chmod";

   function Base return String is
     (if Ada.Environment_Variables.Exists ("ASPIDA_UARP_URL")
      then Ada.Environment_Variables.Value ("ASPIDA_UARP_URL")
      else "https://snaga.ai");

   --  small process-lifetime cache of accepted keys -> user id
   Max_Cache : constant := 256;
   type Entry_T is record K, U : Unbounded_String; end record;
   Cache : array (1 .. Max_Cache) of Entry_T;
   N_Cache : Natural := 0;
   Seq : Natural := 0;   -- unique temp suffix

   function Cached (Key : String; Identity : out Unbounded_String) return Boolean is
   begin
      for I in 1 .. N_Cache loop
         if Cache (I).K = Key then Identity := Cache (I).U; return True; end if;
      end loop;
      return False;
   end Cached;

   procedure Remember (Key : String; User : Unbounded_String) is
   begin
      if N_Cache < Max_Cache then
         N_Cache := N_Cache + 1;
         Cache (N_Cache) := (K => To_Unbounded_String (Key), U => User);
      end if;
   end Remember;

   procedure Del (P : String) is
   begin
      Ada.Directories.Delete_File (P);
   exception when others => null;
   end Del;

   procedure Verify
     (Key : String; Ok : out Boolean; Identity : out Unbounded_String)
   is
      use GNAT.OS_Lib;
      Cfg, Body_F, Code_F : Unbounded_String;
   begin
      Ok := False; Identity := Null_Unbounded_String;
      if Cached (Key, Identity) then Ok := True; return; end if;
      if not Available then return; end if;

      Seq := Seq + 1;
      declare S : constant String := Ada.Strings.Fixed.Trim (Natural'Image (Seq), Ada.Strings.Left); begin
         Cfg    := To_Unbounded_String ("/tmp/aspida_uarp_cfg_" & S);
         Body_F := To_Unbounded_String ("/tmp/aspida_uarp_body_" & S);
         Code_F := To_Unbounded_String ("/tmp/aspida_uarp_code_" & S);
      end;

      --  write the auth header to a 0600 config file (key NOT in argv)
      declare
         F  : Ada.Text_IO.File_Type;
         CS : chars_ptr := New_String (To_String (Cfg));
         RC : int;
         pragma Unreferenced (RC);
      begin
         Ada.Text_IO.Create (F, Ada.Text_IO.Out_File, To_String (Cfg));
         Ada.Text_IO.Close (F);
         RC := c_chmod (CS, 8#600#); Free (CS);
         Ada.Text_IO.Open (F, Ada.Text_IO.Out_File, To_String (Cfg));
         Ada.Text_IO.Put_Line (F, "header = ""Authorization: Bearer " & Key & """");
         Ada.Text_IO.Close (F);
      end;

      declare
         Args : Argument_List :=
           [new String'("-s"), new String'("-K"), new String'(To_String (Cfg)),
            new String'("-o"), new String'(To_String (Body_F)),
            new String'("-w"), new String'("%{http_code}"),
            new String'("--max-time"), new String'("20"),
            new String'(Base & "/api/v1/me")];
         OkS : Boolean; RC : Integer;
      begin
         Spawn (Curl.all, Args, To_String (Code_F), OkS, RC, Err_To_Out => True);
         for A of Args loop Free (A); end loop;
      end;

      --  read HTTP status + body
      declare
         function Slurp (P : String) return String is
            F : Ada.Text_IO.File_Type; R : Unbounded_String;
         begin
            Ada.Text_IO.Open (F, Ada.Text_IO.In_File, P);
            while not Ada.Text_IO.End_Of_File (F) loop
               Append (R, Ada.Text_IO.Get_Line (F));
            end loop;
            Ada.Text_IO.Close (F);
            return To_String (R);
         exception when others => if Ada.Text_IO.Is_Open (F) then Ada.Text_IO.Close (F); end if; return "";
         end Slurp;
         Code : constant String := Slurp (To_String (Code_F));
         Bod  : constant String := Slurp (To_String (Body_F));
      begin
         if Ada.Strings.Fixed.Index (Code, "200") > 0 then
            --  parse "user_id":"..."
            declare
               Tag : constant String := """tenant_id"":""";
               P   : constant Natural := Ada.Strings.Fixed.Index (Bod, Tag);
            begin
               if P > 0 then
                  declare
                     St : constant Natural := P + Tag'Length;
                     E  : Natural := St;
                  begin
                     while E <= Bod'Last and then Bod (E) /= '"' loop E := E + 1; end loop;
                     Identity := To_Unbounded_String (Bod (St .. E - 1));
                  end;
               end if;
               Ok := True;
               Remember (Key, Identity);
            end;
         end if;
      end;

      Del (To_String (Cfg)); Del (To_String (Body_F)); Del (To_String (Code_F));
   end Verify;

end Platform_Auth;
