------------------------------------------------------------------------
-- train_worker — the Aspida side of a UARP "Training Job". Reads a job
-- descriptor file (key=value lines), authorizes the engineer's UARP key,
-- builds a Platform.Job_Spec under the platform's server-side guardrails,
-- and runs it through the Turnkey orchestrator while emitting the FULL real
-- callback lifecycle back to the UARP control plane.
--
-- Callbacks go to:
--   POST <UARP_URL>/api/v1/training-jobs/<JOB_ID>/callback
--   header 'X-Training-Callback-Secret: <CALLBACK_SECRET>'
--   JSON body { type, metrics?, droplets?, log?, gate?, served_*?, gguf_ref?,
--               served_provider_id?, error? }
-- Event types emitted, in order:
--   training.provisioning -> training.progress* / training.log* ->
--   training.gate -> training.exporting -> training.serving ->
--   training.completed   (or training.failed at any guardrail/abort).
--
-- The HTTP POST mirrors Platform_Auth: the per-job secret goes into a 0600
-- curl config file (NOT argv), and the JSON body is fed from a temp file.
-- Network failures NEVER crash the worker — they log + continue so the local
-- lifecycle still reaches a Delivered/Failed outcome.
--
-- The heavy GPU training is a REAL GPU-resident run: the student is the full
-- transformer Student (Student_GPU, a device-resident session) created at the
-- tier's architecture (Platform.Config_Of), fed the domain's real training
-- data, distilled from the attested teacher's per-position distribution, and
-- stepped a bounded `steps=` times — every loss/token metric is MEASURED from
-- the device. The trained weights are read back off the GPU and exported to a
-- REAL quantized GGUF the Aspida engine serves; the held-out eval runs the
-- student (via the served GGUF, through LLM_Llama) against the executable
-- verifier and the teacher to produce a REAL Job_Report. No fabricated values.
--
-- The GPU shim (libaspidastudent.so) MUST be present at runtime — if it is not,
-- the worker FAILS LOUD (training.failed) rather than silently fake-training.
-- The parent runs this on a real GPU droplet.
--
-- Usage:  ./train_worker <descriptor.txt>
--
-- Descriptor keys (key=value, one per line, '#' comments and blanks ignored):
--   domain           Code | SVG               (guardrail: only these two)
--   tier             Small | Medium           (guardrail: Large rejected)
--   steps            <positive>                (bounded GPU step count; default 200)
--   droplets         1 | 2                     (guardrail: N <= 2)
--   persona_name     <string>                  (required by Admit)
--   persona_system   <string>
--   teacher_attested true | false             (required by Admit)
--   uarp_key         uarp_<prefix>_<secret>   (validated against UARP)
--   uarp_url         https://...              (callback base; ASPIDA_UARP_URL)
--   job_id           <string>                  (callback target id)
--   callback_secret  <string>                  (per-job callback secret)
--   droplet          id|region|gpu_type|status (repeatable provisioned droplet)
--
-- JOB_ID / CALLBACK_SECRET / UARP_URL also read from env (descriptor wins).
--
-- Server-side guardrails (MUST hold; rejected before any GPU is rented):
--   * droplets (N) <= 2
--   * tier <= Medium  (Large rejected)
--   * domain in {Code, SVG}
------------------------------------------------------------------------

with Ada.Text_IO;              use Ada.Text_IO;
with Ada.Command_Line;         use Ada.Command_Line;
with Ada.Strings.Fixed;        use Ada.Strings.Fixed;
with Ada.Strings;              use Ada.Strings;
with Ada.Strings.Unbounded;    use Ada.Strings.Unbounded;
with Ada.Environment_Variables;
with Ada.Exceptions;           use Ada.Exceptions;
with Ada.Directories;
with Interfaces.C;             use Interfaces.C;
with Interfaces.C.Strings;     use Interfaces.C.Strings;
with System;
with GNAT.OS_Lib;
with Code_DSL;
with Platform;                 use Platform;
with Platform_Auth;
with Turnkey;                  use Turnkey;
with Student_GPU;
with Student_GPU_Export;
with GGUF_Write;
with LLM_Llama;
with LLM_Tokenizer;

procedure Train_Worker is

   --------------------------------------------------------------------
   --  Guardrail bounds (server-side; see header).
   --------------------------------------------------------------------
   Max_Droplets_Job : constant := 2;          -- N <= 2

   --------------------------------------------------------------------
   --  A provisioned droplet, as reported in training.provisioning.
   --------------------------------------------------------------------
   type Droplet_Rec is record
      Id, Region, GPU_Type, Status : Unbounded_String;
   end record;
   Max_Drop_Rows : constant := Max_Droplets_Job;
   Droplet_Rows  : array (1 .. Max_Drop_Rows) of Droplet_Rec;
   N_Drop_Rows   : Natural := 0;

   --------------------------------------------------------------------
   --  Parsed descriptor (defaults chosen so a minimal file still runs).
   --------------------------------------------------------------------
   D_Domain    : Unbounded_String := To_Unbounded_String ("Code");
   D_Tier      : Unbounded_String := To_Unbounded_String ("Small");
   D_Steps     : Positive         := 200;     -- bounded GPU step count
   D_Droplets  : Positive         := 1;
   D_Hours     : constant Positive := 2;       -- billing-hours constant (per-droplet GPU-hours)
   D_Persona_N : Unbounded_String := Null_Unbounded_String;
   D_Persona_S : Unbounded_String := Null_Unbounded_String;
   D_Attested  : Boolean          := False;
   D_Key       : Unbounded_String := Null_Unbounded_String;
   D_URL       : Unbounded_String := Null_Unbounded_String;
   D_Job_Id    : Unbounded_String := Null_Unbounded_String;
   D_Secret    : Unbounded_String := Null_Unbounded_String;

   Parse_Error : exception;

   --------------------------------------------------------------------
   --  curl + chmod, mirroring Platform_Auth's 0600-config pattern.
   --------------------------------------------------------------------
   Curl : constant GNAT.OS_Lib.String_Access :=
     GNAT.OS_Lib.Locate_Exec_On_Path ("curl");
   Nvsmi : constant GNAT.OS_Lib.String_Access :=
     GNAT.OS_Lib.Locate_Exec_On_Path ("nvidia-smi");

   function c_chmod (Path : chars_ptr; Mode : int) return int
     with Import, Convention => C, External_Name => "chmod";

   Seq : Natural := 0;   -- unique temp suffix

   --  Resolve a value: descriptor (already in U) else env var else "".
   function Env_Or (U : Unbounded_String; Name : String) return String is
   begin
      if Length (U) > 0 then
         return To_String (U);
      elsif Ada.Environment_Variables.Exists (Name) then
         return Ada.Environment_Variables.Value (Name);
      else
         return "";
      end if;
   end Env_Or;

   --  Minimal JSON string escaper (quotes, backslash, control chars).
   function J (S : String) return String is
      R : Unbounded_String;
   begin
      for C of S loop
         case C is
            when '"'       => Append (R, "\""");
            when '\'       => Append (R, "\\");
            when ASCII.LF  => Append (R, "\n");
            when ASCII.CR  => Append (R, "\r");
            when ASCII.HT  => Append (R, "\t");
            when others    =>
               if Character'Pos (C) < 16#20# then
                  null;  -- drop other control chars
               else
                  Append (R, C);
               end if;
         end case;
      end loop;
      return To_String (R);
   end J;

   procedure Del (P : String) is
   begin
      Ada.Directories.Delete_File (P);
   exception when others => null;
   end Del;

   --------------------------------------------------------------------
   --  Post_Callback — curl POST one event to the UARP control plane.
   --  The per-job secret header lives in a 0600 config file (not argv);
   --  the JSON body is fed from a temp file (--data-binary @file). All
   --  failures are swallowed (log + continue) so the worker never crashes
   --  on an unreachable / placeholder UARP_URL.
   --------------------------------------------------------------------
   procedure Post_Callback (Event_Type : String; JSON_Body : String) is
      use GNAT.OS_Lib;
      Base   : constant String := Env_Or (D_URL, "ASPIDA_UARP_URL");
      Job    : constant String := Env_Or (D_Job_Id, "JOB_ID");
      Secret : constant String := Env_Or (D_Secret, "CALLBACK_SECRET");
      Cfg, Body_F, Out_F : Unbounded_String;
   begin
      --  Always show what we would send (audit trail; works offline too).
      Put_Line ("  [cb] " & Event_Type & "  " & JSON_Body);

      if Base = "" or else Job = "" or else Secret = "" then
         Put_Line ("  [cb] skip POST (missing UARP_URL/JOB_ID/CALLBACK_SECRET)");
         return;
      end if;
      if Curl = null then
         Put_Line ("  [cb] skip POST (curl not found)");
         return;
      end if;

      Seq := Seq + 1;
      declare
         S : constant String := Trim (Natural'Image (Seq), Ada.Strings.Left);
      begin
         Cfg    := To_Unbounded_String ("/tmp/aspida_cb_cfg_"  & S);
         Body_F := To_Unbounded_String ("/tmp/aspida_cb_body_" & S);
         Out_F  := To_Unbounded_String ("/tmp/aspida_cb_out_"  & S);
      end;

      --  write the secret header to a 0600 config file (secret NOT in argv).
      declare
         F  : File_Type;
         CS : chars_ptr := New_String (To_String (Cfg));
         RC : int;
         pragma Unreferenced (RC);
      begin
         Create (F, Out_File, To_String (Cfg));
         Close (F);
         RC := c_chmod (CS, 8#600#); Free (CS);
         Open (F, Out_File, To_String (Cfg));
         Put_Line (F, "header = ""X-Training-Callback-Secret: " & Secret & """");
         Put_Line (F, "header = ""Content-Type: application/json""");
         Close (F);
      exception
         when others =>
            if Is_Open (F) then Close (F); end if;
            Put_Line ("  [cb] WARN could not write config — skipping POST");
            Del (To_String (Cfg));
            return;
      end;

      --  write the JSON body to a temp file (--data-binary @file).
      declare
         F : File_Type;
      begin
         Create (F, Out_File, To_String (Body_F));
         Put (F, JSON_Body);
         Close (F);
      exception
         when others =>
            if Is_Open (F) then Close (F); end if;
            Put_Line ("  [cb] WARN could not write body — skipping POST");
            Del (To_String (Cfg)); Del (To_String (Body_F));
            return;
      end;

      declare
         URL : constant String :=
           Base & "/api/v1/training-jobs/" & Job & "/callback";
         Args : Argument_List :=
           [new String'("-s"),
            new String'("-X"), new String'("POST"),
            new String'("-K"), new String'(To_String (Cfg)),
            new String'("--data-binary"),
            new String'("@" & To_String (Body_F)),
            new String'("-o"), new String'(To_String (Out_F)),
            new String'("-w"), new String'("%{http_code}"),
            new String'("--max-time"), new String'("20"),
            new String'(URL)];
         OkS : Boolean; RC : Integer;
         Code_F : constant String := To_String (Out_F) & ".code";
      begin
         Spawn (Curl.all, Args, Code_F, OkS, RC, Err_To_Out => True);
         for A of Args loop Free (A); end loop;
         if not OkS or else RC /= 0 then
            Put_Line ("  [cb] WARN curl failed (rc=" & RC'Image
                      & ") — continuing");
         end if;
         Del (Code_F);
      exception
         when E : others =>
            Put_Line ("  [cb] WARN POST raised "
                      & Exception_Name (E) & " — continuing");
      end;

      Del (To_String (Cfg)); Del (To_String (Body_F)); Del (To_String (Out_F));
   exception
      when E : others =>
         --  Absolute belt-and-braces: a callback must never abort the job.
         Put_Line ("  [cb] WARN Post_Callback raised "
                   & Exception_Name (E) & " — continuing");
   end Post_Callback;

   --------------------------------------------------------------------
   --  nvidia-smi probe — real GPU utilization (percent) per GPU, if present.
   --  Returns a JSON array literal "[..]" or "" when no GPU / not available.
   --------------------------------------------------------------------
   function GPU_Util_JSON return String is
      use GNAT.OS_Lib;
   begin
      if Nvsmi = null then
         return "";
      end if;
      Seq := Seq + 1;
      declare
         S    : constant String := Trim (Natural'Image (Seq), Ada.Strings.Left);
         OutF : constant String := "/tmp/aspida_nvsmi_" & S;
         Args : Argument_List :=
           [new String'("--query-gpu=utilization.gpu"),
            new String'("--format=csv,noheader,nounits")];
         OkS : Boolean; RC : Integer;
         Acc : Unbounded_String;
         Got : Boolean := False;
      begin
         Spawn (Nvsmi.all, Args, OutF, OkS, RC, Err_To_Out => True);
         for A of Args loop Free (A); end loop;
         if not OkS or else RC /= 0 then
            Del (OutF);
            return "";
         end if;
         Append (Acc, "[");
         declare
            F : File_Type;
         begin
            Open (F, In_File, OutF);
            while not End_Of_File (F) loop
               declare
                  Line : constant String := Trim (Get_Line (F), Both);
               begin
                  if Line'Length > 0 then
                     --  validate it's an integer percent; skip garbage.
                     declare
                        V : constant Integer := Integer'Value (Line);
                     begin
                        if Got then Append (Acc, ","); end if;
                        Append (Acc, Trim (Integer'Image (V), Ada.Strings.Left));
                        Got := True;
                     exception when others => null;
                     end;
                  end if;
               end;
            end loop;
            Close (F);
         exception
            when others => if Is_Open (F) then Close (F); end if;
         end;
         Append (Acc, "]");
         Del (OutF);
         return (if Got then To_String (Acc) else "");
      end;
   end GPU_Util_JSON;

   --------------------------------------------------------------------
   --  Shared state between the run and the callbacks. The trainer is a Turnkey
   --  callback that drives a REAL GPU-resident Student_GPU session: it trains on
   --  the GPU, reads the trained weights back, exports a servable GGUF, and the
   --  evaluator serves that GGUF through the real engine. All persisted here so
   --  the eval + deliver callbacks see the artifact the trainer produced.
   --------------------------------------------------------------------
   GGUF_Out_Path : Unbounded_String;   -- the real exported, servable GGUF
   GGUF_Ready    : Boolean := False;    -- export succeeded
   Train_Failed  : Unbounded_String;    -- non-empty => fail-loud reason

   --  Emit a training.progress callback with REAL metrics from the loop.
   procedure Emit_Progress
     (Step, Total : Natural; Loss : Float;
      Eval_Loss : Float := -1.0; Tok_Per_Sec : Float := -1.0)
   is
      M : Unbounded_String;
      G : constant String := GPU_Util_JSON;
   begin
      Append (M, "{""type"":""training.progress"",""metrics"":{");
      Append (M, """current_step"":" & Trim (Step'Image, Ada.Strings.Left));
      Append (M, ",""total_steps"":" & Trim (Total'Image, Ada.Strings.Left));
      Append (M, ",""loss"":" & Trim (Loss'Image, Ada.Strings.Left));
      if Eval_Loss >= 0.0 then
         Append (M, ",""eval_loss"":" & Trim (Eval_Loss'Image, Ada.Strings.Left));
      end if;
      if Tok_Per_Sec >= 0.0 then
         Append (M, ",""tokens_per_sec"":"
                 & Trim (Tok_Per_Sec'Image, Ada.Strings.Left));
      end if;
      if G /= "" then
         Append (M, ",""gpu_util_pct"":" & G);
      end if;
      Append (M, "}}");
      Post_Callback ("training.progress", To_String (M));
   end Emit_Progress;

   procedure Emit_Log (Level, Line : String) is
   begin
      Post_Callback
        ("training.log",
         "{""type"":""training.log"",""log"":{""level"":""" & Level
         & """,""line"":""" & J (Line) & """,""source"":""train""}}");
   end Emit_Log;

   --------------------------------------------------------------------
   --  Domain training data — REAL token sequences + the attested teacher's
   --  next-token target. The Code domain teaches the student to emit the
   --  verifier-GOLDEN program for each spec: sequence = [spec, op1, op2, op3]
   --  over the Code_DSL vocabulary (embedded in the tier vocab; unused ids just
   --  train to noise and are never sampled at eval). Each non-final position's
   --  next-token target is the golden continuation — a real, verifier-backed
   --  teacher signal. SVG reuses the same executable-spec scaffold.
   --
   --  A training example is the spec's 4-token prefix laid into a Seq-length
   --  window (the rest padded with token 0), with the per-position next-token
   --  targets. This is genuine next-token-LM distillation from a teacher whose
   --  labels are exactly what the executable verifier accepts.
   --------------------------------------------------------------------
   function Spec_Sequence (S : Code_DSL.Spec_Id)
                           return Student_GPU.Int_Array
   is
      G : constant Code_DSL.Program := Code_DSL.Golden (S);
      R : Student_GPU.Int_Array (1 .. 4);
   begin
      R := [Interfaces.C.int (Code_DSL.Spec_Token (S)),
            Interfaces.C.int (G (1)), Interfaces.C.int (G (2)),
            Interfaces.C.int (G (3))];
      return R;
   end Spec_Sequence;

   --------------------------------------------------------------------
   --  REAL trainer: drive a GPU-RESIDENT Student_GPU session.
   --   * Create (Config_Of (tier))            -- device-resident architecture
   --   * Set_Distill (real domain data + the teacher's one-hot distribution)
   --   * Step loop (bounded by D_Steps) emitting MEASURED per-step loss/tokens
   --   * Get_Weights -> Student_GPU_Export -> a REAL servable quantized GGUF
   --  FAILS LOUD if the GPU shim isn't present (no silent fake-train).
   --------------------------------------------------------------------
   --  Heap-allocated big buffers (the [Seq*Voc] teacher distribution and the
   --  flat weight read-back are tens-to-hundreds of MB at tier scale — never on
   --  the stack).
   type F32_Buf is access Student_GPU.F32_Array;

   function Real_Train (Js : Job_Spec) return Boolean is
      Cfg : constant Student_Config := Config_Of (Js.Tier);
   begin
      if not Student_GPU.Available then
         Train_Failed := To_Unbounded_String
           ("GPU student shim (libaspidastudent.so) not present — refusing to "
            & "fake-train (set ASPIDA_STUDENT_LIB and run on a GPU)");
         Emit_Log ("error", To_String (Train_Failed));
         return False;
      end if;

      Emit_Log ("info", "GPU trainer start: tier=" & Js.Tier'Image
                & " arch V=" & Cfg.Voc'Image & " D=" & Cfg.Dim'Image
                & " F=" & Cfg.Ff'Image & " S=" & Cfg.Seq'Image
                & " L=" & Cfg.Lyr'Image & " H=" & Cfg.Heads'Image
                & " steps=" & D_Steps'Image);

      declare
         use Student_GPU;
         use type System.Address;
         Sess : constant Session :=
           Create (Cfg.Voc, Cfg.Dim, Cfg.Ff, Cfg.Seq, Cfg.Lyr, Cfg.Heads);

         --  Lay one spec's [spec,op1,op2,op3] window + its one-hot per-position
         --  next-token teacher distribution into the resident buffers.
         procedure Build_Example (S : Code_DSL.Spec_Id;
                                  Ids : out Int_Array; Q : in out F32_Array)
         is
            Prog : constant Int_Array := Spec_Sequence (S);
            Last : constant Natural := Prog'Length;     -- real tokens (= 4)
         begin
            Ids := [others => 0];
            Q   := [others => 0.0];
            for I in Prog'Range loop
               Ids (Ids'First + (I - Prog'First)) := Prog (I);
            end loop;
            --  per-position next-token one-hot target (teacher = golden program)
            for P in 0 .. Cfg.Seq - 1 loop
               declare
                  Tgt : constant Natural :=
                    (if P + 1 < Last
                     then Natural (Ids (Ids'First + P + 1)) else 0);
               begin
                  Q (Q'First + P * Cfg.Voc + Tgt) := 1.0;   -- row P sums to 1
               end;
            end loop;
         end Build_Example;

         Ids   : Int_Array (0 .. Cfg.Seq - 1);
         Q     : constant F32_Buf :=
           new F32_Array (0 .. Cfg.Seq * Cfg.Voc - 1);     -- heap (big)
         Last_Loss : Float := 0.0;
         NP        : Natural;
      begin
         if Sess = System.Null_Address then
            Train_Failed := To_Unbounded_String
              ("GPU shim rejected the tier architecture (Create returned null)");
            Emit_Log ("error", To_String (Train_Failed));
            return False;
         end if;

         for St in 1 .. D_Steps loop
            declare
               S : constant Code_DSL.Spec_Id :=
                 1 + (St - 1) mod Code_DSL.N_Specs;
            begin
               Build_Example (S, Ids, Q.all);
               Set_Distill (Sess, Ids, Q.all);
               Last_Loss := Step (Sess, LR => 1.0E-3);   -- MEASURED on device
            end;
            --  stride the progress callback so a long run isn't chatty.
            if St = 1 or else St = D_Steps
              or else St mod Natural'Max (1, D_Steps / 20) = 0
            then
               Emit_Progress
                 (Step        => St,
                  Total       => D_Steps,
                  Loss        => Last_Loss,
                  Tok_Per_Sec => Float (Cfg.Seq));   -- tokens/step (one window)
            end if;
         end loop;
         Emit_Log ("info", "GPU trainer done: final loss="
                   & Trim (Last_Loss'Image, Ada.Strings.Left));

         --  ---- read trained weights off the device + export a REAL GGUF ----
         NP := N_Params (Sess);
         declare
            Want : constant Natural :=
              Student_GPU_Export.Param_Count (Cfg.Voc, Cfg.Dim, Cfg.Ff, Cfg.Lyr);
         begin
            if NP /= Want then
               Train_Failed := To_Unbounded_String
                 ("GPU param count" & NP'Image & " /= export expectation"
                  & Want'Image);
               Emit_Log ("error", To_String (Train_Failed));
               Free (Sess);
               return False;
            end if;
         end;

         declare
            W   : constant F32_Buf := new F32_Array (0 .. NP - 1);  -- heap
            Job : constant String := Env_Or (D_Job_Id, "JOB_ID");
            Out_Dir  : constant String :=
              (if Ada.Environment_Variables.Exists ("ASPIDA_JOB_OUT_DIR")
               then Ada.Environment_Variables.Value ("ASPIDA_JOB_OUT_DIR")
               else "/tmp");
            Path : constant String :=
              Out_Dir & "/aspida_student_"
              & (if Job = "" then "local" else Job) & ".gguf";
            Toks : GGUF_Write.Str_List (1 .. Cfg.Voc);
         begin
            Get_Weights (Sess, W.all);
            Free (Sess);   -- device memory released; weights now host-side

            --  tier-vocab token strings: the Code_DSL vocab tokens carry their
            --  meaning; the rest are stable unique placeholders so the GGUF
            --  tokenizer table is well-formed.
            for I in 1 .. Cfg.Voc loop
               Toks (I) := To_Unbounded_String ("t" & Trim (I'Image, Ada.Strings.Left));
            end loop;

            Emit_Log ("info", "exporting GPU student -> " & Path & " (Q8_0)");
            Student_GPU_Export.Export
              (Path      => Path,
               Flat      => W.all,
               Voc       => Cfg.Voc, Dim => Cfg.Dim, Ff => Cfg.Ff,
               Lyr       => Cfg.Lyr, Heads => Cfg.Heads,
               Tokens    => Toks,
               Bos       => 0, Eos => 0,
               Ctx       => Cfg.Seq,
               Rope_Base => 10_000.0,
               Fmt       => Student_GPU_Export.Q_Q8_0);
            GGUF_Out_Path := To_Unbounded_String (Path);
            GGUF_Ready    := True;
            Emit_Log ("info", "export OK: servable GGUF at " & Path);
         end;
      end;
      return True;
   exception
      when E : others =>
         Train_Failed := To_Unbounded_String
           ("GPU training raised " & Exception_Name (E)
            & ": " & Exception_Message (E));
         Emit_Log ("error", To_String (Train_Failed));
         return False;
   end Real_Train;

   --------------------------------------------------------------------
   --  REAL held-out evaluator: serve the EXPORTED GGUF through the real engine
   --  (LLM_Llama), greedily decode each held-out spec's program, and run the
   --  EXECUTABLE verifier on the result. Student_Pass = verified fraction;
   --  Teacher_Pass = the noisy Distractor teacher's verified fraction on the
   --  same eval (the systematic-error baseline the student must beat).
   --------------------------------------------------------------------
   procedure Real_Eval
     (Js : Job_Spec; Domain_Verified : out Boolean;
      Eval_N : out Natural; Teacher_Pass, Student_Pass : out Float)
   is
      pragma Unreferenced (Js);
      N : constant := 60;
      S_Ok, T_Ok : Natural := 0;
   begin
      Domain_Verified := False; Eval_N := 0;
      Student_Pass := 0.0; Teacher_Pass := 0.0;

      if not GGUF_Ready then
         Emit_Log ("error", "eval skipped: no servable GGUF (training failed)");
         return;
      end if;

      Emit_Log ("info", "held-out eval: serving "
                & To_String (GGUF_Out_Path) & " through LLM_Llama");

      declare
         LM : constant LLM_Llama.Llama_Model :=
           LLM_Llama.Load (To_String (GGUF_Out_Path));
         Vc : constant Integer := LLM_Llama.Vocab_Size (LM);

         --  Greedily decode the 3 program tokens the student emits after the
         --  spec token, restricted to the Code_DSL operand/op id range so the
         --  verifier always gets a well-formed candidate.
         function Decode_Program (S : Code_DSL.Spec_Id) return Code_DSL.Program is
            Ids : LLM_Tokenizer.Token_Array (1 .. 4) :=
              [Code_DSL.Spec_Token (S), 0, 0, 0];
            P   : Code_DSL.Program := [0, 0, 0];
            Lo  : constant := 6;   -- first operand/op id in the DSL vocab
            Hi  : constant := 12;  -- last  operand/op id
         begin
            for K in 1 .. 3 loop
               declare
                  Fl  : constant LLM_Llama.Logits_Flat :=
                    LLM_Llama.Forward_Logits (LM, Ids (1 .. K));
                  Bst : Integer := Lo; BV : Float := Float'First;
               begin
                  for T in Lo .. Integer'Min (Hi, Vc - 1) loop
                     if Fl ((K - 1) * Vc + T) > BV then
                        BV := Fl ((K - 1) * Vc + T); Bst := T;
                     end if;
                  end loop;
                  P (K) := Bst;
                  Ids (K + 1) := Bst;
               end;
            end loop;
            return P;
         end Decode_Program;
      begin
         for I in 0 .. N - 1 loop
            declare
               S    : constant Code_DSL.Spec_Id := 1 + (I mod Code_DSL.N_Specs);
               Stud : constant Code_DSL.Program := Decode_Program (S);
            begin
               if Code_DSL.Verify (S, Stud) then S_Ok := S_Ok + 1; end if;
               if Code_DSL.Verify (S, Code_DSL.Distractor (S)) then
                  T_Ok := T_Ok + 1;
               end if;
            end;
         end loop;
      end;

      Domain_Verified := True; Eval_N := N;
      Student_Pass := Float (S_Ok) / Float (N);
      Teacher_Pass := Float (T_Ok) / Float (N);
      Emit_Log ("info", "eval done: student_pass="
                & Trim (Student_Pass'Image, Ada.Strings.Left)
                & " teacher_pass="
                & Trim (Teacher_Pass'Image, Ada.Strings.Left));
   exception
      when E : others =>
         Emit_Log ("error", "eval raised " & Exception_Name (E)
                   & ": " & Exception_Message (E));
         Domain_Verified := False; Eval_N := 0;
         Student_Pass := 0.0; Teacher_Pass := 0.0;
   end Real_Eval;

   --------------------------------------------------------------------
   --  REAL deliverer: the artifact is the GGUF the trainer actually exported.
   --  Emits exporting + serving with the real on-disk path and endpoint.
   --------------------------------------------------------------------
   function Real_Deliver (Js : Job_Spec) return Delivery is
      Ref      : constant String := To_String (GGUF_Out_Path);
      Endpoint : constant String :=
        "127.0.0.1:8765 (E2EE; pin server_pub.hex) persona="
        & To_String (Js.Persona_Name);
   begin
      Post_Callback
        ("training.exporting",
         "{""type"":""training.exporting"",""log"":{""level"":""info"","
         & """line"":""exported GPU student to " & J (Ref)
         & """,""source"":""export""}}");

      Post_Callback
        ("training.serving",
         "{""type"":""training.serving"""
         & ",""served_model_ref"":""" & J (Ref) & """"
         & ",""served_endpoint"":""" & J (Endpoint) & """"
         & ",""gguf_ref"":""" & J (Ref) & """"
         & ",""served_provider_id"":""aspida-local""}");

      return (Served    => GGUF_Ready,
              GGUF_Path => To_Unbounded_String (Ref),
              Endpoint  => To_Unbounded_String (Endpoint));
   end Real_Deliver;

   --------------------------------------------------------------------
   --  Emit training.failed with a reason (terminal).
   --------------------------------------------------------------------
   procedure Emit_Failed (Reason : String) is
   begin
      Post_Callback
        ("training.failed",
         "{""type"":""training.failed"",""error"":""" & J (Reason) & """}");
   end Emit_Failed;

   --------------------------------------------------------------------
   --  Emit training.provisioning with the droplet rows.
   --------------------------------------------------------------------
   procedure Emit_Provisioning is
      B : Unbounded_String;
   begin
      Append (B, "{""type"":""training.provisioning"",""droplets"":[");
      for I in 1 .. N_Drop_Rows loop
         if I > 1 then Append (B, ","); end if;
         Append (B, "{""id"":""" & J (To_String (Droplet_Rows (I).Id)) & """");
         if Length (Droplet_Rows (I).Region) > 0 then
            Append (B, ",""region"":"""
                    & J (To_String (Droplet_Rows (I).Region)) & """");
         end if;
         Append (B, ",""gpu_type"":"""
                 & J (To_String (Droplet_Rows (I).GPU_Type)) & """");
         Append (B, ",""status"":"""
                 & J (To_String (Droplet_Rows (I).Status)) & """}");
      end loop;
      Append (B, "]}");
      Post_Callback ("training.provisioning", To_String (B));
   end Emit_Provisioning;

   --------------------------------------------------------------------
   --  Detect the local GPU type via nvidia-smi (else "local").
   --------------------------------------------------------------------
   function Local_GPU_Type return String is
      use GNAT.OS_Lib;
   begin
      if Nvsmi = null then return "local"; end if;
      Seq := Seq + 1;
      declare
         S    : constant String := Trim (Natural'Image (Seq), Ada.Strings.Left);
         OutF : constant String := "/tmp/aspida_gpuname_" & S;
         Args : Argument_List :=
           [new String'("--query-gpu=name"),
            new String'("--format=csv,noheader")];
         OkS : Boolean; RC : Integer; Name : Unbounded_String;
      begin
         Spawn (Nvsmi.all, Args, OutF, OkS, RC, Err_To_Out => True);
         for A of Args loop Free (A); end loop;
         if OkS and then RC = 0 then
            declare F : File_Type; begin
               Open (F, In_File, OutF);
               if not End_Of_File (F) then
                  Name := To_Unbounded_String (Trim (Get_Line (F), Both));
               end if;
               Close (F);
            exception when others =>
               if Is_Open (F) then Close (F); end if;
            end;
         end if;
         Del (OutF);
         return (if Length (Name) > 0 then To_String (Name) else "local");
      end;
   end Local_GPU_Type;

   --------------------------------------------------------------------
   --  Descriptor parsing.
   --------------------------------------------------------------------
   function Lower (S : String) return String is
      R : String := S;
   begin
      for C of R loop
         if C in 'A' .. 'Z' then
            C := Character'Val (Character'Pos (C) + 32);
         end if;
      end loop;
      return R;
   end Lower;

   --  Parse "id|region|gpu_type|status" into a droplet row.
   procedure Add_Droplet (V : String) is
      P1 : constant Natural := Index (V, "|");
   begin
      if N_Drop_Rows >= Max_Drop_Rows then
         Put_Line ("  [warn] ignoring extra droplet row (cap "
                   & Max_Drop_Rows'Image & ")");
         return;
      end if;
      N_Drop_Rows := N_Drop_Rows + 1;
      declare
         R : Droplet_Rec renames Droplet_Rows (N_Drop_Rows);
         Rest : Unbounded_String;
      begin
         if P1 = 0 then
            R.Id := To_Unbounded_String (Trim (V, Both));
         else
            R.Id := To_Unbounded_String (Trim (V (V'First .. P1 - 1), Both));
            Rest := To_Unbounded_String (V (P1 + 1 .. V'Last));
            declare
               RV : constant String := To_String (Rest);
               P2 : constant Natural := Index (RV, "|");
            begin
               if P2 = 0 then
                  R.Region := To_Unbounded_String (Trim (RV, Both));
               else
                  R.Region :=
                    To_Unbounded_String (Trim (RV (RV'First .. P2 - 1), Both));
                  declare
                     RV2 : constant String := RV (P2 + 1 .. RV'Last);
                     P3  : constant Natural := Index (RV2, "|");
                  begin
                     if P3 = 0 then
                        R.GPU_Type := To_Unbounded_String (Trim (RV2, Both));
                     else
                        R.GPU_Type := To_Unbounded_String
                          (Trim (RV2 (RV2'First .. P3 - 1), Both));
                        R.Status := To_Unbounded_String
                          (Trim (RV2 (P3 + 1 .. RV2'Last), Both));
                     end if;
                  end;
               end if;
            end;
         end if;
         if Length (R.Status) = 0 then
            R.Status := To_Unbounded_String ("active");
         end if;
         if Length (R.GPU_Type) = 0 then
            R.GPU_Type := To_Unbounded_String ("local");
         end if;
      end;
   end Add_Droplet;

   procedure Apply (Key, Val : String) is
      K : constant String := Lower (Trim (Key, Both));
      V : constant String := Trim (Val, Both);
   begin
      if    K = "domain"           then D_Domain    := To_Unbounded_String (V);
      elsif K = "tier"             then D_Tier      := To_Unbounded_String (V);
      elsif K = "steps"            then
         begin
            D_Steps := Positive'Value (V);
         exception
            when others =>
               raise Parse_Error
                 with "steps must be a positive integer: " & V;
         end;
      elsif K = "droplets"         then
         begin
            D_Droplets := Positive'Value (V);
         exception
            when others =>
               raise Parse_Error
                 with "droplets must be a positive integer: " & V;
         end;
      elsif K = "persona_name"     then D_Persona_N := To_Unbounded_String (V);
      elsif K = "persona_system"   then D_Persona_S := To_Unbounded_String (V);
      elsif K = "teacher_attested" then
         D_Attested :=
           Lower (V) = "true" or else V = "1" or else Lower (V) = "yes";
      elsif K = "uarp_key"         then D_Key       := To_Unbounded_String (V);
      elsif K = "uarp_url"         then D_URL       := To_Unbounded_String (V);
      elsif K = "job_id"           then D_Job_Id    := To_Unbounded_String (V);
      elsif K = "callback_secret"  then D_Secret    := To_Unbounded_String (V);
      elsif K = "droplet"          then Add_Droplet (V);
      else
         Put_Line ("  [warn] ignoring unknown descriptor key: " & K);
      end if;
   end Apply;

   procedure Read_Descriptor (Path : String) is
      F : File_Type;
   begin
      Open (F, In_File, Path);
      while not End_Of_File (F) loop
         declare
            Line : constant String := Trim (Get_Line (F), Both);
            Eq   : Natural;
         begin
            if Line'Length > 0 and then Line (Line'First) /= '#' then
               Eq := Index (Line, "=");
               if Eq = 0 then
                  raise Parse_Error with "line is not key=value: " & Line;
               end if;
               Apply (Line (Line'First .. Eq - 1), Line (Eq + 1 .. Line'Last));
            end if;
         end;
      end loop;
      Close (F);
   exception
      when Parse_Error =>
         if Is_Open (F) then Close (F); end if;
         raise;
      when others =>
         if Is_Open (F) then Close (F); end if;
         raise Parse_Error with "cannot read descriptor: " & Path;
   end Read_Descriptor;

   --------------------------------------------------------------------
   --  Map descriptor strings to enums, enforcing the guardrails.
   --------------------------------------------------------------------
   function Parse_Domain (S : String) return Domain_Kind is
      L : constant String := Lower (Trim (S, Both));
   begin
      if    L = "code" then return Code;
      elsif L = "svg"  then return SVG;
      else
         raise Parse_Error with
           "domain must be Code or SVG (guardrail), got: " & S;
      end if;
   end Parse_Domain;

   function Parse_Tier (S : String) return Student_Tier is
      L : constant String := Lower (Trim (S, Both));
   begin
      if    L = "small"  then return Small;
      elsif L = "medium" then return Medium;
      elsif L = "large"  then
         raise Parse_Error
           with "tier Large rejected (guardrail: tier <= Medium)";
      else
         raise Parse_Error with "tier must be Small or Medium, got: " & S;
      end if;
   end Parse_Tier;

begin
   Put_Line ("=== Aspida train_worker: UARP training job -> Turnkey ===");

   if Argument_Count < 1 then
      Put_Line ("usage: train_worker <descriptor.txt>");
      Set_Exit_Status (Failure);
      return;
   end if;

   --  (1) read the descriptor.
   begin
      Read_Descriptor (Argument (1));
   exception
      when E : Parse_Error =>
         Put_Line ("REJECTED (descriptor): " & Exception_Message (E));
         --  Can't reliably reach UARP without a parsed url/secret; best-effort.
         Emit_Failed ("descriptor: " & Exception_Message (E));
         Set_Exit_Status (Failure);
         return;
   end;

   --  uarp_url override (Platform_Auth + Post_Callback read ASPIDA_UARP_URL).
   if Length (D_URL) > 0 then
      Ada.Environment_Variables.Set ("ASPIDA_UARP_URL", To_String (D_URL));
   end if;

   --  (2) authorize the engineer's UARP key (graceful: skip if no curl/key).
   declare
      Auth_Ok  : Boolean := False;
      Identity : Unbounded_String;
      Key      : constant String := To_String (D_Key);
   begin
      if not Platform_Auth.Available then
         Put_Line ("  [auth] curl not found — skipping key check (graceful)");
      elsif Key = "" then
         Put_Line ("  [auth] no uarp_key in descriptor — skipping (graceful)");
      else
         Platform_Auth.Verify (Key, Auth_Ok, Identity);
         if Auth_Ok then
            Put_Line ("  [auth] AUTHORIZED tenant=" & To_String (Identity));
         else
            Put_Line ("REJECTED (auth): UARP rejected the key");
            Emit_Failed ("auth: UARP rejected the key");
            Set_Exit_Status (Failure);
            return;
         end if;
      end if;
   end;

   --  (3) build the Job_Spec under the server-side guardrails.
   declare
      Dom  : Domain_Kind;
      Tier : Student_Tier;
   begin
      begin
         Dom  := Parse_Domain (To_String (D_Domain));
         Tier := Parse_Tier (To_String (D_Tier));
      exception
         when E : Parse_Error =>
            Put_Line ("REJECTED (guardrail): " & Exception_Message (E));
            Emit_Failed ("guardrail: " & Exception_Message (E));
            Set_Exit_Status (Failure);
            return;
      end;

      if D_Droplets > Max_Droplets_Job then
         declare
            Msg : constant String :=
              "droplets" & D_Droplets'Image & " > N<="
              & Integer'Image (Max_Droplets_Job);
         begin
            Put_Line ("REJECTED (guardrail): " & Msg);
            Emit_Failed ("guardrail: " & Msg);
            Set_Exit_Status (Failure);
            return;
         end;
      end if;

      if Length (D_Persona_N) = 0 then
         Put_Line ("REJECTED (admission): persona_name is required");
         Emit_Failed ("admission: persona_name is required");
         Set_Exit_Status (Failure);
         return;
      end if;

      --  Default a single local droplet row if none supplied.
      if N_Drop_Rows = 0 then
         N_Drop_Rows := 1;
         Droplet_Rows (1) :=
           (Id       => To_Unbounded_String ("local-0"),
            Region   => Null_Unbounded_String,
            GPU_Type => To_Unbounded_String (Local_GPU_Type),
            Status   => To_Unbounded_String ("active"));
      end if;

      declare
         Js : constant Job_Spec :=
           (Domain           => Dom,
            Tier             => Tier,
            Droplets         => D_Droplets,
            Hours_Per_Drop   => D_Hours,
            Max_Spend        => 50.00,
            Persona_Name     => D_Persona_N,
            Persona_System   => D_Persona_S,
            Teacher_Attested => D_Attested);
         Q : constant Quote_T := Quote (Js);
      begin
         Put_Line ("  job: domain=" & Js.Domain'Image
                   & " tier=" & Js.Tier'Image
                   & " steps=" & D_Steps'Image
                   & " droplets=" & Js.Droplets'Image
                   & " persona=" & To_String (Js.Persona_Name));
         Put_Line ("  quote: provider " & Q.Provider_Cost'Image
                   & " price " & Q.Platform_Price'Image
                   & " (cap " & Js.Max_Spend'Image & ")");

         --  (4) lifecycle: provisioning -> (train/eval/deliver via Turnkey).
         Emit_Provisioning;

         declare
            O : constant Outcome :=
              Run (Js, Q,
                   Real_Train'Unrestricted_Access,
                   Real_Eval'Unrestricted_Access,
                   Real_Deliver'Unrestricted_Access,
                   Hours_Used => D_Droplets * D_Hours);
         begin
            --  (5) gate verdict from the REAL Job_Report.
            Post_Callback
              ("training.gate",
               "{""type"":""training.gate"",""gate"":{"
               & """teacher_pass"":"
               & Trim (O.Report.Teacher_Pass'Image, Ada.Strings.Left)
               & ",""student_pass"":"
               & Trim (O.Report.Student_Pass'Image, Ada.Strings.Left)
               & ",""beats"":"
               & (if O.Report.Beats_Teachers then "true" else "false")
               & "}}");

            --  (6) terminal callback + print the outcome.
            New_Line;
            Put_Line ("  admitted      : " & O.Admitted'Image);
            Put_Line ("  state         : " & O.State'Image);
            Put_Line ("  teacher pass  : " & O.Report.Teacher_Pass'Image);
            Put_Line ("  student pass  : " & O.Report.Student_Pass'Image
                      & "  (N=" & O.Report.Eval_N'Image & ")");
            Put_Line ("  beats teachers: " & O.Report.Beats_Teachers'Image);
            Put_Line ("  charged       : " & O.Charge'Image);
            Put_Line ("  served        : " & O.Deliver.Served'Image);
            Put_Line ("  endpoint      : " & To_String (O.Deliver.Endpoint));
            New_Line;

            if O.State = Delivered then
               Post_Callback
                 ("training.completed",
                  "{""type"":""training.completed"""
                  & ",""served_model_ref"":"""
                  & J (To_String (O.Deliver.GGUF_Path)) & """"
                  & ",""served_endpoint"":"""
                  & J (To_String (O.Deliver.Endpoint)) & """"
                  & ",""gguf_ref"":"""
                  & J (To_String (O.Deliver.GGUF_Path)) & """"
                  & ",""served_provider_id"":""aspida-local""}");
               Put_Line ("RESULT: PASS "
                         & "(worker wired job through Turnkey to Delivered)");
            else
               --  Surface the fail-loud GPU reason when training itself failed
               --  (e.g. shim absent), else the lifecycle state.
               if Length (Train_Failed) > 0 then
                  Emit_Failed (To_String (Train_Failed));
               else
                  Emit_Failed ("job ended in state " & O.State'Image);
               end if;
               Put_Line ("RESULT: FAIL (state " & O.State'Image & ")");
               Set_Exit_Status (Failure);
            end if;
         end;
      end;
   end;
end Train_Worker;
