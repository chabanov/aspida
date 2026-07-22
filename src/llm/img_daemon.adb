with GNAT.Sockets;              use GNAT.Sockets;
with Ada.Environment_Variables;

package body Img_Daemon is

   --  Localhost TCP port aspida-imgd listens on (ASPIDA_IMGD_PORT, default 8790).
   function Daemon_Port return Port_Type is
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_IMGD_PORT") then
         return Port_Type'Value
           (Ada.Environment_Variables.Value ("ASPIDA_IMGD_PORT"));
      end if;
      return 8790;
   exception
      when others => return 8790;
   end Daemon_Port;

   --  Strip the leading space Integer'Image emits for non-negative values.
   function Num (S : String) return String is
     (if S'Length > 0 and then S (S'First) = ' '
      then S (S'First + 1 .. S'Last) else S);

   function Generate
     (Prompt   : String;
      Ref_Path : String := "";
      Width    : Integer := 1024;
      Height   : Integer := 1024;
      Steps    : Integer := 20;
      Seed     : Long_Long_Integer := -1;
      Out_Path : String)
      return Boolean
   is
      --  Header line: "<W> <H> <steps> <seed> <plen> <rlen> <olen>\n"
      --  then the raw prompt, ref-path and out-path bytes (length-prefixed so
      --  the prompt may contain any character).
      Header : constant String :=
        Num (Integer'Image (Width))  & " " &
        Num (Integer'Image (Height)) & " " &
        Num (Integer'Image (Steps))  & " " &
        Num (Long_Long_Integer'Image (Seed)) & " " &
        Num (Integer'Image (Prompt'Length))   & " " &
        Num (Integer'Image (Ref_Path'Length)) & " " &
        Num (Integer'Image (Out_Path'Length)) & Character'Val (10);
      Sock : Socket_Type;
      Ok   : Boolean := False;
   begin
      Create_Socket (Sock);
      --  Bound the wait: image gen can take ~30-70s; a hung daemon must NOT
      --  wedge secure_server, so cap the receive at 300s -> timeout raises,
      --  we catch it and return False.
      Set_Socket_Option
        (Sock, Socket_Level, (Receive_Timeout, Timeout => 300.0));
      Connect_Socket
        (Sock, (Family_Inet, Inet_Addr ("127.0.0.1"), Daemon_Port));
      declare
         S    : constant Stream_Access := Stream (Sock);
         Line : String (1 .. 64);
         N    : Natural := 0;
         C    : Character;
      begin
         String'Write (S, Header);
         String'Write (S, Prompt);
         String'Write (S, Ref_Path);
         String'Write (S, Out_Path);
         --  Read the single reply line ("OK" / "ERR <rc>").
         loop
            Character'Read (S, C);
            exit when C = Character'Val (10);
            if N < Line'Last then
               N := N + 1;
               Line (N) := C;
            end if;
         end loop;
         Ok := N >= 2 and then Line (1 .. 2) = "OK";
      end;
      Close_Socket (Sock);
      return Ok;
   exception
      when others =>
         begin
            Close_Socket (Sock);
         exception
            when others => null;
         end;
         return False;
   end Generate;

end Img_Daemon;
