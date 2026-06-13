---------------------------------------------------------------------
-- LLM_Qwen_Blk body
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_RMSNorm;
with LLM_RoPE;
with LLM_Qwen_Attn;
with LLM_MoE;
with LLM_SSM;
with Ada.Numerics.Elementary_Functions;

package body LLM_Qwen_Blk is

   use LLM_Tensor;
   use Ada.Numerics.Elementary_Functions;

   function Create_Qwen_Block
     (QKV_W           : Tensor;
      Attn_Gate_W     : Tensor;
      O_W             : Tensor;
      Attn_Norm_W     : Tensor;
      Post_Attn_Norm_W : Tensor;
      Ssm_Params      : LLM_SSM.SSM_Params;
      Moe_Layer       : LLM_MoE.MoE_Layer;
      Is_Full_Attn    : Boolean;
      Dim             : Integer;
      N_Heads         : Integer;
      N_KV_Heads      : Integer)
      return Qwen_Block
   is
      B : Qwen_Block;
   begin
      B.Is_Full_Attn := Is_Full_Attn;
      B.QKV_W := QKV_W;
      B.Attn_Gate_W := Attn_Gate_W;
      B.O_W := O_W;
      B.Attn_Norm_W := Attn_Norm_W;
      B.Post_Attn_Norm_W := Post_Attn_Norm_W;
      B.SSM := Ssm_Params;
      B.MoE := Moe_Layer;
      B.Dim := Dim;
      B.N_Heads := N_Heads;
      B.N_KV_Heads := N_KV_Heads;
      return B;
   end Create_Qwen_Block;

   function Forward (B : Qwen_Block; X : Tensor) return Tensor is
      H : Tensor := X;
      Dim : constant Integer := B.Dim;
      RoPE : constant LLM_RoPE.RoPE_Params := LLM_RoPE.Create_Qwen_RoPE;
      Head_Dim : constant Integer := RoPE.Dim;  -- 64
   begin
      -- Step 1: RMSNorm + Attention/SSM
      declare
         Norm_X : constant Tensor := LLM_RMSNorm.Forward (H, B.Attn_Norm_W);
         Attn_Out : Tensor;
      begin
         if B.Is_Full_Attn then
            -- Full attention layer
            Attn_Out := LLM_Qwen_Attn.Forward (
              LLM_Qwen_Attn.Create_Qwen_Attn_Params (
                B.QKV_W, B.O_W, B.Attn_Gate_W, RoPE,
                Dim, B.N_Heads, B.N_KV_Heads, Head_Dim,
                False), -- no gate on full-attn layers
              Norm_X, 0);
         else
            -- SSM + sliding attention with gate
            declare
               Attn : constant Tensor := LLM_Qwen_Attn.Forward (
                 LLM_Qwen_Attn.Create_Qwen_Attn_Params (
                   B.QKV_W, B.O_W, B.Attn_Gate_W, RoPE,
                   Dim, B.N_Heads, B.N_KV_Heads, Head_Dim,
                   True),
                 Norm_X, 0);
               Ssm_State : Tensor := LLM_SSM.Init_State (128);  -- state_dim = 128
               Ssm_Out   : constant Tensor := LLM_SSM.Forward (B.SSM, Norm_X, Ssm_State);
               -- Gate: sigmoid(gate) * attn + (1-sigmoid(gate)) * ssm
               Gate_Sum : Float := 0.0;
               Gate_Val : Float;
            begin
               for I in 1 .. Dim loop
                  Gate_Sum := Gate_Sum + Get_Flat (Norm_X, I)
                    * Get (B.Attn_Gate_W, [I, 1]);
               end loop;
               Gate_Val := 1.0 / (1.0 + Ada.Numerics.Elementary_Functions.Exp (-Gate_Sum));

               Attn_Out := New_Tensor ([1, Dim]);
               for I in 1 .. Dim loop
                  Set_Flat (Attn_Out, I,
                    Gate_Val * Get_Flat (Attn, I) +
                    (1.0 - Gate_Val) * Get_Flat (Ssm_Out, I));
               end loop;
            end;
         end if;

         -- Residual connection
         for I in 1 .. Dim loop
            Set_Flat (H, I, Get_Flat (H, I) + Get_Flat (Attn_Out, I));
         end loop;
      end;

      -- Step 2: Post-attention norm + MoE
      declare
         Norm_H : constant Tensor := LLM_RMSNorm.Forward (H, B.Post_Attn_Norm_W);
         Moe_Out : constant Tensor := LLM_MoE.Forward (B.MoE, Norm_H);
      begin
         for I in 1 .. Dim loop
            Set_Flat (H, I, Get_Flat (H, I) + Get_Flat (Moe_Out, I));
         end loop;
      end;

      return H;
   end Forward;

   function "=" (Left, Right : Qwen_Block) return Boolean is
   begin
      return Left.Is_Full_Attn = Right.Is_Full_Attn
        and Left.Dim = Right.Dim
        and Left.N_Heads = Right.N_Heads
        and Left.N_KV_Heads = Right.N_KV_Heads;
   end "=";

end LLM_Qwen_Blk;
