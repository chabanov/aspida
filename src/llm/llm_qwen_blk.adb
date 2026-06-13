---------------------------------------------------------------------
-- LLM_Qwen_Blk body
---------------------------------------------------------------------

with LLM_RMSNorm;
with LLM_RoPE;
with LLM_Qwen_Attn;
with Ada.Numerics.Elementary_Functions;

package body LLM_Qwen_Blk is

   use LLM_Tensor;

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
      Seq      : constant Integer := Shape (X) (1);
      Dim      : constant Integer := B.Dim;
      RoPE     : constant LLM_RoPE.RoPE_Params := LLM_RoPE.Create_Qwen_RoPE;
      Head_Dim : constant Integer := RoPE.Dim;  -- 64
      H        : Tensor := X;                    -- working state [Seq, Dim]

      --  Extract row T of a [Seq, Dim] tensor as a [1, Dim] vector.
      function Row (M : Tensor; T : Integer) return Tensor is
         R : Tensor := New_Tensor ([1, Dim]);
      begin
         for I in 1 .. Dim loop
            Set_Flat (R, I, Get (M, [T, I]));
         end loop;
         return R;
      end Row;
   begin
      ----------------------------------------------------------------
      -- Step 1: pre-attention RMSNorm (per row) + attention / SSM blend
      ----------------------------------------------------------------
      declare
         Norm_X   : Tensor := New_Tensor ([Seq, Dim]);
         Attn_Out : Tensor := New_Tensor ([Seq, Dim]);
      begin
         for T in 1 .. Seq loop
            declare
               NR : constant Tensor := LLM_RMSNorm.Forward (Row (H, T), B.Attn_Norm_W);
            begin
               for I in 1 .. Dim loop
                  Set (Norm_X, [T, I], Get_Flat (NR, I));
               end loop;
            end;
         end loop;

         --  Causal self-attention over the whole sequence. No internal gate:
         --  gating, if any, is applied at the block level below.
         declare
            Attn : constant Tensor := LLM_Qwen_Attn.Forward (
              LLM_Qwen_Attn.Create_Qwen_Attn_Params (
                B.QKV_W, B.O_W, B.Attn_Gate_W, RoPE,
                Dim, B.N_Heads, B.N_KV_Heads, Head_Dim, False),
              Norm_X, 0);
         begin
            if B.Is_Full_Attn then
               Attn_Out := Attn;
            else
               --  SSM layer: run the recurrent SSM over the sequence and blend
               --  it with attention via a per-row sigmoid gate.
               declare
                  State : Tensor := LLM_SSM.Init_State (128);  -- state_dim = 128
               begin
                  for T in 1 .. Seq loop
                     declare
                        NR       : constant Tensor := Row (Norm_X, T);
                        Ssm_Row  : constant Tensor := LLM_SSM.Forward (B.SSM, NR, State);
                        Gate_Sum : Float := 0.0;
                        Gate_Val : Float;
                     begin
                        for I in 1 .. Dim loop
                           Gate_Sum := Gate_Sum
                             + Get (Norm_X, [T, I]) * Get (B.Attn_Gate_W, [I, 1]);
                        end loop;
                        Gate_Val :=
                          1.0 / (1.0 + Ada.Numerics.Elementary_Functions.Exp (-Gate_Sum));
                        for I in 1 .. Dim loop
                           Set (Attn_Out, [T, I],
                             Gate_Val * Get (Attn, [T, I])
                             + (1.0 - Gate_Val) * Get_Flat (Ssm_Row, I));
                        end loop;
                     end;
                  end loop;
               end;
            end if;
         end;

         --  Residual connection
         for T in 1 .. Seq loop
            for I in 1 .. Dim loop
               Set (H, [T, I], Get (H, [T, I]) + Get (Attn_Out, [T, I]));
            end loop;
         end loop;
      end;

      ----------------------------------------------------------------
      -- Step 2: post-attention RMSNorm (per row) + MoE (per row) + residual
      ----------------------------------------------------------------
      for T in 1 .. Seq loop
         declare
            NR      : constant Tensor :=
              LLM_RMSNorm.Forward (Row (H, T), B.Post_Attn_Norm_W);
            Moe_Row : constant Tensor := LLM_MoE.Forward (B.MoE, NR);
         begin
            for I in 1 .. Dim loop
               Set (H, [T, I], Get (H, [T, I]) + Get_Flat (Moe_Row, I));
            end loop;
         end;
      end loop;

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
