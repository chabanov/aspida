---------------------------------------------------------------------
-- LLM_Weight body
---------------------------------------------------------------------

with LLM_Dequant;

package body LLM_Weight is

   use LLM_Tensor;

   function From_Dense (T : Tensor) return Weight is
   begin
      return W : Weight do
         W.Present_F := True;
         W.Is_Quant  := False;
         W.Dense     := T;
      end return;
   end From_Dense;

   function From_Quant
     (Info : LLM_GGUF.Tensor_Info; Bytes : Byte_Data) return Weight is
   begin
      return W : Weight do
         W.Present_F := True;
         W.Is_Quant  := True;
         W.Info      := Info;
         W.Bytes     := Bytes;
      end return;
   end From_Quant;

   function Present (W : Weight) return Boolean is (W.Present_F);

   function Rows (W : Weight) return Integer is
   begin
      if W.Is_Quant then
         return Integer (W.Info.Dims (2));
      else
         return Shape (W.Dense) (1);
      end if;
   end Rows;

   function Cols (W : Weight) return Integer is
   begin
      if W.Is_Quant then
         return Integer (W.Info.Dims (1));
      else
         return Shape (W.Dense) (2);
      end if;
   end Cols;

   function Count (W : Weight) return Long_Long_Integer is
      C : Long_Long_Integer := 1;
   begin
      if not W.Present_F then
         return 0;
      elsif W.Is_Quant then
         for D in 1 .. Natural (W.Info.N_Dims) loop
            C := C * Long_Long_Integer (W.Info.Dims (D));
         end loop;
         return C;
      else
         return Long_Long_Integer (Numel (W.Dense));
      end if;
   end Count;

   function Dense_MatVec (D, X : Tensor) return Tensor is
      Out_Dim : constant Integer := Shape (D) (1);
      In_Dim  : constant Integer := Shape (D) (2);
   begin
      return Y : Tensor := New_Tensor ([1, Out_Dim]) do
         for O in 1 .. Out_Dim loop
            declare
               Acc : Float := 0.0;
            begin
               for I in 1 .. In_Dim loop
                  Acc := Acc + Get (D, [O, I]) * Get_Flat (X, I);
               end loop;
               Set_Flat (Y, O, Acc);
            end;
         end loop;
      end return;
   end Dense_MatVec;

   function MatVec (W : Weight; X : Tensor) return Tensor is
   begin
      if W.Is_Quant then
         return LLM_Dequant.QMatVec (W.Info, W.Bytes.all, X);
      else
         return Dense_MatVec (W.Dense, X);
      end if;
   end MatVec;

   function N_Experts (W : Weight) return Integer is
   begin
      if W.Is_Quant then
         return Integer (W.Info.Dims (3));
      else
         return Shape (W.Dense) (1);
      end if;
   end N_Experts;

   function Expert_Out (W : Weight) return Integer is
   begin
      if W.Is_Quant then
         return Integer (W.Info.Dims (2));
      else
         return Shape (W.Dense) (2);
      end if;
   end Expert_Out;

   function MatVec_Expert (W : Weight; E : Integer; X : Tensor) return Tensor is
   begin
      if W.Is_Quant then
         declare
            Sub : LLM_GGUF.Tensor_Info := W.Info;
            BPE : Natural;
         begin
            Sub.N_Dims := 2;
            Sub.Dims   := [W.Info.Dims (1), W.Info.Dims (2), 0, 0];
            BPE := Natural (LLM_GGUF.Tensor_Byte_Size (Sub));
            declare
               Start : constant Natural := W.Bytes.all'First + (E - 1) * BPE;
            begin
               return LLM_Dequant.QMatVec
                 (Sub, W.Bytes (Start .. Start + BPE - 1), X);
            end;
         end;
      else
         --  Dense 3D [n_exp, out_e, in].
         declare
            Out_E : constant Integer := Shape (W.Dense) (2);
            In_D  : constant Integer := Shape (W.Dense) (3);
         begin
            return Y : Tensor := New_Tensor ([1, Out_E]) do
               for O in 1 .. Out_E loop
                  declare
                     Acc : Float := 0.0;
                  begin
                     for I in 1 .. In_D loop
                        Acc := Acc + Get (W.Dense, [E, O, I]) * Get_Flat (X, I);
                     end loop;
                     Set_Flat (Y, O, Acc);
                  end;
               end loop;
            end return;
         end;
      end if;
   end MatVec_Expert;

end LLM_Weight;
