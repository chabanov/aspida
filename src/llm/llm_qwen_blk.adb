---------------------------------------------------------------------
-- LLM_Qwen_Blk body
---------------------------------------------------------------------

with LLM_RMSNorm;

package body LLM_Qwen_Blk is

   use LLM_Tensor;

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
            Moe_Row : constant Tensor := LLM_MoE.Forward (B.MoE, NR);
         begin
            for I in 1 .. Dim loop
               Set (H, [T, I], Get (H, [T, I]) + Get_Flat (Moe_Row, I));
            end loop;
         end;
      end loop;

      return H;
   end Forward;

end LLM_Qwen_Blk;
