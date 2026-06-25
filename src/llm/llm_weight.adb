---------------------------------------------------------------------
-- LLM_Weight body
---------------------------------------------------------------------

with Ada.Environment_Variables;
with Ada.Text_IO;
with LLM_Dequant;

package body LLM_Weight is

   use LLM_Tensor;

   --------------------------------------------------------------------
   -- Dense weight cache (OPT-IN — off by default, enable with LLM_WEIGHT_CACHE)
   --
   -- During generation the same 2D weights are re-dequantized from Q5_K/
   -- Q6_K on every token. This caches their dense F32 form on first use and
   -- serves subsequent matvecs via the fast flat MatVec_Rows — no per-token
   -- decode. Only 2D weights are cached (the always-used attention / delta /
   -- shared-expert / router projections, ~5.3 GB); the 256-per-layer routed
   -- experts stay quantized (decoded on demand). Result is bit-identical to
   -- QMatVec (same Fill dequant, same ascending dot order); lookup/fill run
   -- only on the main thread (experts use the 3D MatVec_Expert), so no lock.
   --
   -- MEASURED: on this Apple-silicon CPU it is a small *regression* — the
   -- fused QMatVec decode is already cheap on NEON, and the dense F32 set is
   -- 4x the Q5_K bytes, so it thrashes the CPU caches and costs more RSS
   -- (~+5.3 GB) than the decode it saves. Kept opt-in because it can pay off
   -- where decode dominates (slower-decode formats, or a future backend).
   --------------------------------------------------------------------

   Cache_Enabled : constant Boolean :=
     Ada.Environment_Variables.Exists ("LLM_WEIGHT_CACHE");
   Budget_Bytes : constant Long_Long_Integer := 16 * 1024 ** 3;  -- 16 GiB cap

   type Tensor_Access is access LLM_Tensor.Tensor;
   type Cache_Entry is record
      Key : Byte_Data    := null;
      Val : Tensor_Access := null;
   end record;

   Max_Entries : constant := 4096;
   Cache       : array (1 .. Max_Entries) of Cache_Entry;
   N_Cached    : Natural := 0;
   Used_Bytes  : Long_Long_Integer := 0;

   --  Dense [out, in] form of a quantized 2D weight, or null if not cacheable
   --  (cache full / over budget) so the caller falls back to streaming QMatVec.
   function Cached_Dense (W : Weight) return Tensor_Access is
      Sz : constant Long_Long_Integer :=
        Long_Long_Integer (Rows (W)) * Long_Long_Integer (Cols (W)) * 4;
   begin
      for I in 1 .. N_Cached loop
         if Cache (I).Key = W.Bytes then
            return Cache (I).Val;
         end if;
      end loop;
      if N_Cached >= Max_Entries or else Used_Bytes + Sz > Budget_Bytes then
         return null;
      end if;
      declare
         T : constant Tensor_Access :=
           new LLM_Tensor.Tensor'(LLM_Dequant.Dequantize (W.Info, W.Bytes.all));
      begin
         N_Cached := N_Cached + 1;
         Cache (N_Cached) := (Key => W.Bytes, Val => T);
         Used_Bytes := Used_Bytes + Sz;
         if Ada.Environment_Variables.Exists ("LLM_CACHE_DEBUG") then
            Ada.Text_IO.Put_Line
              ("  [wcache] #" & Natural'Image (N_Cached)
               & " kind=" & LLM_GGUF.GGML_Type'Image (W.Info.Kind)
               & " " & Integer'Image (Rows (W)) & " x" & Integer'Image (Cols (W))
               & "  total_MB=" & Long_Long_Integer'Image (Used_Bytes / 1024 / 1024));
         end if;
         return T;
      end;
   end Cached_Dense;

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

   function Get_Row (W : Weight; Row : Integer) return Tensor is
   begin
      if not W.Is_Quant then
         declare
            In_Dim : constant Integer := Shape (W.Dense) (2);
            R : Tensor := New_Tensor ([1, In_Dim]);
         begin
            for I in 1 .. In_Dim loop
               Set_Flat (R, I, Get (W.Dense, [Row + 1, I]));
            end loop;
            return R;
         end;
      end if;
      declare
         Row_Info : LLM_GGUF.Tensor_Info := W.Info;
      begin
         Row_Info.N_Dims := 2;
         Row_Info.Dims   := [W.Info.Dims (1), 1, 0, 0];
         --  Synthetic 1-row slice: clear the inherited full-tensor Byte_Size so
         --  Tensor_Byte_Size recomputes it from the 1-row dims.
         Row_Info.Byte_Size := 0;
         declare
            BPR : constant Natural :=
              Natural (LLM_GGUF.Tensor_Byte_Size (Row_Info));
            RS  : constant Natural := W.Bytes'First + Row * BPR;
         begin
            return LLM_Dequant.Dequantize
              (Row_Info, W.Bytes (RS .. RS + BPR - 1));
         end;
      end;
   end Get_Row;

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
         if Cache_Enabled and then Natural (W.Info.N_Dims) = 2 then
            declare
               D : constant Tensor_Access := Cached_Dense (W);
            begin
               if D /= null then
                  return MatVec_Rows (D.all, X);
               end if;
            end;
         end if;
         return LLM_Dequant.QMatVec (W.Info, W.Bytes.all, X);
      else
         return Dense_MatVec (W.Dense, X);
      end if;
   end MatVec;

   function Raw_Address (W : Weight) return System.Address is
   begin
      if W.Is_Quant and then W.Bytes /= null then
         return W.Bytes.all'Address;
      else
         return System.Null_Address;
      end if;
   end Raw_Address;

   function Raw_Bytes (W : Weight) return Long_Long_Integer is
   begin
      if W.Is_Quant and then W.Bytes /= null then
         return Long_Long_Integer (W.Bytes.all'Length);
      else
         return 0;
      end if;
   end Raw_Bytes;

   function Kind_Code (W : Weight) return Integer is
      use type LLM_GGUF.GGML_Type;
   begin
      if not W.Is_Quant then
         return -1;
      end if;
      case W.Info.Kind is
         when LLM_GGUF.GGML_TYPE_Q4_K => return 0;
         when LLM_GGUF.GGML_TYPE_Q6_K => return 1;
         when LLM_GGUF.GGML_TYPE_Q5_K => return 2;
         when LLM_GGUF.GGML_TYPE_Q3_K => return 3;
         when LLM_GGUF.GGML_TYPE_Q2_K => return 4;
         when others => return -1;
      end case;
   end Kind_Code;

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
            --  Per-expert 2D slice: clear the inherited full-tensor Byte_Size
            --  so Tensor_Byte_Size recomputes it from the per-expert dims.
            Sub.Byte_Size := 0;
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
