---------------------------------------------------------------------
-- LLM_Qwen_Blk body
---------------------------------------------------------------------

with LLM_RMSNorm;

package body LLM_Qwen_Blk is

   use LLM_Tensor;

   --  Dispatch the post-attention FFN: routed MoE or dense SwiGLU.
   function FFN (B : Qwen_Block; X : Tensor) return Tensor is
   begin
      if B.Is_MoE then
         return LLM_MoE.Forward (B.MoE, X);
      else
         return LLM_Dense_FFN.Forward (B.Dense, X);
      end if;
   end FFN;

   function Forward (B : Qwen_Block; X : Tensor) return Tensor is
      Seq : constant Integer := Shape (X) (1);
      Dim : constant Integer := B.Dim;
      H   : Tensor := X;   -- residual stream [seq, dim]

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
      -- Step 1: pre-attention RMSNorm (per row) + attention / delta-net
      ----------------------------------------------------------------
      declare
         Norm_X   : Tensor := New_Tensor ([Seq, Dim]);
         Attn_Out : Tensor;
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

         if B.Is_Full_Attn then
            Attn_Out := LLM_FullAttn.Forward (B.Full, Norm_X);
         else
            Attn_Out := LLM_DeltaNet_Blk.Forward (B.DNet, Norm_X);
         end if;

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
            Moe_Row : constant Tensor := FFN (B, NR);
         begin
            for I in 1 .. Dim loop
               Set (H, [T, I], Get (H, [T, I]) + Get_Flat (Moe_Row, I));
            end loop;
         end;
      end loop;

      return H;
   end Forward;

   --------------------------------------------------------------------
   -- Incremental decode
   --------------------------------------------------------------------

   function Init_State (B : Qwen_Block; Max_Len : Integer) return Block_State is
   begin
      return St : Block_State do
         St.Is_Full := B.Is_Full_Attn;
         if B.Is_Full_Attn then
            St.Full_St := LLM_FullAttn.Init_State (B.Full, Max_Len);
         else
            St.DNet_St := LLM_DeltaNet_Blk.Init_State (B.DNet);
         end if;
      end return;
   end Init_State;

   function Step (B : Qwen_Block; St : in out Block_State; X : Tensor)
      return Tensor
   is
      Dim      : constant Integer := B.Dim;
      H        : Tensor := X;   -- residual stream [1, dim]
      Norm_X   : constant Tensor := LLM_RMSNorm.Forward (X, B.Attn_Norm_W);
      Attn_Out : Tensor;
   begin
      --  Step 1: pre-attention RMSNorm + (attention | delta-net) + residual.
      if B.Is_Full_Attn then
         Attn_Out := LLM_FullAttn.Step (B.Full, St.Full_St, Norm_X);
      else
         Attn_Out := LLM_DeltaNet_Blk.Step (B.DNet, St.DNet_St, Norm_X);
      end if;
      for I in 1 .. Dim loop
         Set_Flat (H, I, Get_Flat (H, I) + Get_Flat (Attn_Out, I));
      end loop;

      --  Step 2: post-attention RMSNorm + MoE + residual.
      declare
         NR      : constant Tensor := LLM_RMSNorm.Forward (H, B.Post_Attn_Norm_W);
         Moe_Row : constant Tensor := FFN (B, NR);
      begin
         for I in 1 .. Dim loop
            Set_Flat (H, I, Get_Flat (H, I) + Get_Flat (Moe_Row, I));
         end loop;
      end;

      return H;
   end Step;

end LLM_Qwen_Blk;
