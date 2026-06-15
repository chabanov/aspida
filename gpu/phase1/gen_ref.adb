--  Stream B · Phase 1 — reference generator (CPU side).
--  Reads a real Q4_K llama GGUF and emits little-endian binary fixtures the
--  GPU self-test (qk.cu) checks against:
--    dq_in.bin  (144 bytes: one Q4_K super-block of token_embd)
--    dq_exp.bin (256 float32: CPU Dequant_Q4_K of that block)
--    mv_w.bin   (raw Q4_K bytes of blk.0.attn_k.weight, in=2560 out=512)
--    mv_x.bin   (2560 float32: deterministic input vector)
--    mv_y.bin   (512  float32: CPU LLM_Weight.MatVec output)
--
--  Usage:  gen_ref <llama-model.gguf>
with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Tensor;  use LLM_Tensor;
with LLM_Dequant;
with LLM_Weight;
with LLM_RMSNorm;
with LLM_RoPE;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;

procedure Gen_Ref is
   Path : constant String := Ada.Command_Line.Argument (1);
   G    : GGUF_File;

   procedure Write_Bytes (Name : String; S : String) is
      F : File_Type;
   begin
      Create (F, Out_File, Name);
      String'Write (Stream (F), S);
      Close (F);
   end Write_Bytes;

   procedure Write_Floats (Name : String; T : Tensor; N : Integer) is
      F : File_Type;
   begin
      Create (F, Out_File, Name);
      for I in 1 .. N loop
         Float'Write (Stream (F), Get_Flat (T, I));
      end loop;
      Close (F);
   end Write_Floats;
begin
   Open (G, Path);

   --  (1) dequant fixture: token_embd super-block 0.
   declare
      Info : constant Tensor_Info := Find_Tensor (G, "token_embd.weight");
      Buf  : aliased String (1 .. 144);
      Q    : Tensor := New_Tensor ([1, 256]);
   begin
      Read_Tensor_Range (G, Info, 0, Buf'Address, 144);
      LLM_Dequant.Dequant_Q4_K (Buf, Q, 256);
      Write_Bytes  ("dq_in.bin", Buf);
      Write_Floats ("dq_exp.bin", Q, 256);
   end;

   --  (2) matvec fixture: blk.0.attn_k.weight (Q4_K, ne0=2560 in, ne1=512 out).
   declare
      Info : constant Tensor_Info := Find_Tensor (G, "blk.0.attn_k.weight");
      In_D : constant Integer := Integer (Info.Dims (1));
      Out_D : constant Integer := Integer (Info.Dims (2));
      Size : constant Natural := Natural (Tensor_Byte_Size (Info));
      B    : constant LLM_Weight.Byte_Data := new String (1 .. Size);
      X    : Tensor := New_Tensor ([1, In_D]);
   begin
      Read_Tensor_Raw (G, Info, B.all'Address, Size);
      for I in 1 .. In_D loop
         Set_Flat (X, I, Float ((I mod 13) - 6) * 0.1);   -- deterministic
      end loop;
      declare
         W : constant LLM_Weight.Weight := LLM_Weight.From_Quant (Info, B);
         Y : constant Tensor := LLM_Weight.MatVec (W, X);
      begin
         Write_Bytes  ("mv_w.bin", B.all);
         Write_Floats ("mv_x.bin", X, In_D);
         Write_Floats ("mv_y.bin", Y, Out_D);
         Ada.Text_IO.Put_Line ("gen_ref: in=" & In_D'Image & " out=" & Out_D'Image
           & " wbytes=" & Size'Image & "  (kind=" & Info.Kind'Image & ")");
      end;
   end;

   --  (3) RMSNorm fixture (deterministic x, w; eps 1e-6).
   declare
      D : constant Integer := 4096;
      X : Tensor := New_Tensor ([1, D]);
      W : Tensor := New_Tensor ([1, D]);
   begin
      for I in 1 .. D loop
         Set_Flat (X, I, Float (((I * 7) mod 23) - 11) * 0.05);
         Set_Flat (W, I, 0.5 + Float (I mod 5) * 0.1);
      end loop;
      declare
         Y : constant Tensor := LLM_RMSNorm.Forward (X, W);
      begin
         Write_Floats ("rms_x.bin", X, D);
         Write_Floats ("rms_w.bin", W, D);
         Write_Floats ("rms_y.bin", Y, D);
      end;
   end;

   --  (4) RoPE fixture (head_dim 128, base 1e6, real rope_freqs, pos 7).
   declare
      HD  : constant Integer := 128;
      Pos : constant Integer := 7;
      RfI : constant Tensor_Info := Find_Tensor (G, "rope_freqs.weight");
      RfN : constant Natural := Natural (Tensor_Byte_Size (RfI));
      RfB : aliased String (1 .. RfN);
      P   : LLM_RoPE.RoPE_Params :=
        LLM_RoPE.Create_Qwen_RoPE (HD, 500_000.0, 131_072);
      X   : Tensor := New_Tensor ([1, HD]);
   begin
      Read_Tensor_Raw (G, RfI, RfB'Address, RfN);
      declare
         FF : constant Tensor := LLM_Dequant.Dequantize (RfI, RfB);
      begin
         LLM_RoPE.Set_Freq_Factors (P, FF);
         for I in 1 .. HD loop
            Set_Flat (X, I, Float (((I * 3) mod 17) - 8) * 0.1);
         end loop;
         declare
            Y : constant Tensor := LLM_RoPE.Apply (P, X, Pos);
         begin
            Write_Floats ("rope_in.bin",  X,  HD);
            Write_Floats ("rope_ff.bin",  FF, HD / 2);
            Write_Floats ("rope_out.bin", Y,  HD);
         end;
      end;
   end;

   --  (5) Full Llama transformer layer (blk.0) at pos 0 / seq 1: reference
   --  output computed with the validated engine primitives, in Forward_Step
   --  order, so the GPU layer (layer.cu) can be checked against it. RoPE is
   --  identity at pos 0 and single-position attention is softmax=1 -> V, so this
   --  exercises rmsnorm x2, the 7 matvecs, SwiGLU, GQA broadcast and residuals.
   declare
      HD : constant Integer := 128;
      function LF (Name : String) return Tensor is
         I : constant Tensor_Info := Find_Tensor (G, Name);
         N : constant Natural := Natural (Tensor_Byte_Size (I));
         B : aliased String (1 .. N);
      begin
         Read_Tensor_Raw (G, I, B'Address, N);
         return LLM_Dequant.Dequantize (I, B);
      end LF;
      function LZ (Name : String; Bytes : out LLM_Weight.Byte_Data)
                   return LLM_Weight.Weight is
         I : constant Tensor_Info := Find_Tensor (G, Name);
         N : constant Natural := Natural (Tensor_Byte_Size (I));
      begin
         Bytes := new String (1 .. N);
         Read_Tensor_Raw (G, I, Bytes.all'Address, N);
         return LLM_Weight.From_Quant (I, Bytes);
      end LZ;

      BQ, BK, BV, BO, BG, BU, BD : LLM_Weight.Byte_Data;
      AN : constant Tensor := LF ("blk.0.attn_norm.weight");
      FN : constant Tensor := LF ("blk.0.ffn_norm.weight");
      WQ : constant LLM_Weight.Weight := LZ ("blk.0.attn_q.weight", BQ);
      WK : constant LLM_Weight.Weight := LZ ("blk.0.attn_k.weight", BK);
      WV : constant LLM_Weight.Weight := LZ ("blk.0.attn_v.weight", BV);
      WO : constant LLM_Weight.Weight := LZ ("blk.0.attn_output.weight", BO);
      WGt : constant LLM_Weight.Weight := LZ ("blk.0.ffn_gate.weight", BG);
      WUp : constant LLM_Weight.Weight := LZ ("blk.0.ffn_up.weight", BU);
      WDn : constant LLM_Weight.Weight := LZ ("blk.0.ffn_down.weight", BD);
      D   : constant Integer := LLM_Weight.Cols (WQ);     -- 8192
      FFN : constant Integer := LLM_Weight.Rows (WGt);    -- 28672
      KV  : constant Integer := LLM_Weight.Rows (WK);     -- 1024
      NH  : constant Integer := D / HD;
      NKV : constant Integer := KV / HD;
      X   : Tensor := New_Tensor ([1, D]);
   begin
      for I in 1 .. D loop Set_Flat (X, I, Float (((I * 5) mod 19) - 9) * 0.02); end loop;
      declare
         XN  : constant Tensor := LLM_RMSNorm.Forward (X, AN);
         V   : constant Tensor := LLM_Weight.MatVec (WV, XN);
         Ctx : Tensor := New_Tensor ([1, NH * HD]);
         X1  : Tensor := New_Tensor ([1, D]);
      begin
         --  pos0: RoPE identity, single-pos softmax=1 -> ctx = V (GQA broadcast).
         for H in 0 .. NH - 1 loop
            declare KvH : constant Integer := H / (NH / NKV); begin
               for J in 1 .. HD loop
                  Set_Flat (Ctx, H * HD + J, Get_Flat (V, KvH * HD + J));
               end loop;
            end;
         end loop;
         declare
            Attn : constant Tensor := LLM_Weight.MatVec (WO, Ctx);
         begin
            for I in 1 .. D loop Set_Flat (X1, I, Get_Flat (X, I) + Get_Flat (Attn, I)); end loop;
         end;
         declare
            XN2 : constant Tensor := LLM_RMSNorm.Forward (X1, FN);
            Gp  : constant Tensor := LLM_Weight.MatVec (WGt, XN2);
            Up  : constant Tensor := LLM_Weight.MatVec (WUp, XN2);
            GU  : Tensor := New_Tensor ([1, FFN]);
            Y   : Tensor := New_Tensor ([1, D]);
         begin
            for I in 1 .. FFN loop
               declare S : constant Float := Get_Flat (Gp, I); begin
                  Set_Flat (GU, I, (S / (1.0 + Exp (-S))) * Get_Flat (Up, I));
               end;
            end loop;
            declare Dn : constant Tensor := LLM_Weight.MatVec (WDn, GU); begin
               for I in 1 .. D loop Set_Flat (Y, I, Get_Flat (X1, I) + Get_Flat (Dn, I)); end loop;
            end;
            Write_Floats ("layer_x.bin",  X,  D);
            Write_Floats ("layer_an.bin", AN, D);
            Write_Floats ("layer_fn.bin", FN, D);
            Write_Bytes  ("layer_wq.bin", BQ.all); Write_Bytes ("layer_wk.bin", BK.all);
            Write_Bytes  ("layer_wv.bin", BV.all); Write_Bytes ("layer_wo.bin", BO.all);
            Write_Bytes  ("layer_wg.bin", BG.all); Write_Bytes ("layer_wu.bin", BU.all);
            Write_Bytes  ("layer_wd.bin", BD.all);
            Write_Floats ("layer_y.bin",  Y,  D);
            Ada.Text_IO.Put_Line ("layer ref: dim=" & D'Image & " ffn=" & FFN'Image
              & " nh=" & NH'Image & " nkv=" & NKV'Image);
         end;
      end;
   end;

   Close (G);
end Gen_Ref;
