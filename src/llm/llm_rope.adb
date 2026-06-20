---------------------------------------------------------------------
-- LLM_RoPE body — mRoPE implementation for Qwen 3.5
---------------------------------------------------------------------

with Ada.Numerics;
use Ada.Numerics;
with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;

package body LLM_RoPE is

   procedure Set_Interleaved (P : in out RoPE_Params; On : Boolean := True) is
   begin
      P.Interleaved := On;
   end Set_Interleaved;

   function Create_Qwen_RoPE
     (Dim       : Integer := 64;
      Freq_Base : Float   := 10_000_000.0;
      Max_Pos   : Integer := 262_144) return RoPE_Params
   is
      P : RoPE_Params;
      S : Tensor := New_Tensor ([1, 4]);
   begin
      P.Dim       := Dim;
      P.Freq_Base := Freq_Base;
      P.Max_Pos   := Max_Pos;
      Set_Flat (S, 1, 11.0);  -- section 0: dims 0-10
      Set_Flat (S, 2, 11.0);  -- section 1: dims 11-21
      Set_Flat (S, 3, 10.0);  -- section 2: dims 22-31
      Set_Flat (S, 4, 0.0);   -- section 3: unused
      P.Sections := S;
      P.Use_FF := False;
      P.Freq_Factors := New_Tensor ([1, 1]);  -- sentinel (unused)
      return P;
   end Create_Qwen_RoPE;

   procedure Set_Freq_Factors (P : in out RoPE_Params; FF : Tensor) is
   begin
      P.Freq_Factors := FF;
      P.Use_FF := True;
   end Set_Freq_Factors;

   procedure Set_Linear_Scale (P : in out RoPE_Params; Factor : Float) is
   begin
      if Factor > 1.0 then
         P.Freq_Scale := 1.0 / Factor;
      else
         P.Freq_Scale := 1.0;   -- no-op for factor <= 1
      end if;
   end Set_Linear_Scale;

   procedure Set_NTK_Scale (P : in out RoPE_Params; Factor : Float) is
   begin
      if Factor > 1.0 and then P.Dim > 2 then
         --  base' = base * factor^(dim/(dim-2))  (NTK-aware interpolation).
         P.Freq_Base := P.Freq_Base
           * Factor ** (Float (P.Dim) / Float (P.Dim - 2));
      end if;
   end Set_NTK_Scale;

   procedure Set_Yarn_Scale
     (P : in out RoPE_Params; Factor : Float; N_Ctx_Orig : Integer;
      Beta_Fast : Float := 32.0; Beta_Slow : Float := 1.0)
   is
      --  Dimension at which a given number of rotations spans the original
      --  context (llama.cpp ggml_rope_yarn_corr_dim).
      function Corr_Dim (N_Rot : Float) return Float is
        (Float (P.Dim)
         * Log (Float (N_Ctx_Orig) / (N_Rot * 2.0 * Pi))
         / (2.0 * Log (P.Freq_Base)));
   begin
      if Factor <= 1.0 or else N_Ctx_Orig <= 0 then
         return;   -- no-op
      end if;
      P.Freq_Scale := 1.0 / Factor;
      P.Corr_Low  := Float'Max (0.0, Float'Floor (Corr_Dim (Beta_Fast)));
      P.Corr_High := Float'Min (Float (P.Dim - 1),
                                Float'Ceiling (Corr_Dim (Beta_Slow)));
      P.M_Scale   := 1.0 + 0.1 * Log (Factor);   -- attention temperature
      P.Yarn_On   := True;
   end Set_Yarn_Scale;

   function Apply (P : RoPE_Params; X : Tensor; Pos : Integer) return Tensor is
      Half_Dim : constant Integer := P.Dim / 2;  -- 32 for Qwen
      Result   : Tensor := New_Tensor ([1, P.Dim]);
      Theta    : Float;
      Cos_Val  : Float;
      Sin_Val  : Float;
      X1, X2   : Float;
   begin
      --  mRoPE / Sections note: Qwen3.5-MoE mRoPE splits the head into 3
      --  frequency sections (time/height/width, widths 11/11/10 here, stored
      --  in P.Sections). For TEXT-only inference — the only mode this engine
      --  supports — every text token uses t = h = w = Pos, so a single position
      --  applied across all dims is exactly correct and P.Sections needs no
      --  per-section position routing. Multimodal (vision) inputs would need a
      --  separate position per section; that path is not wired in (no image
      --  encoder), so Sections is deliberately a no-op here, not a bug. The
      --  caller (LLM_FullAttn.Norm_Rope) passes one Pos for all dims by design.
      --
      --  NeoX / rotate_half convention (Qwen, Gemma): pair dimension i with
      --  i + dim/2 (first half with second half), NOT adjacent (2i, 2i+1).
      --  Llama instead sets P.Interleaved (see below): adjacent pairs (2i, 2i+1).
      --    theta_i = pos / freq_base^(2i/dim)
      --    out[i]          = x[i]*cos - x[i+d/2]*sin
      --    out[i+d/2]      = x[i+d/2]*cos + x[i]*sin
      for I in 0 .. Half_Dim - 1 loop
         declare
            --  theta_extrap = pos / base^(2i/dim) (raw / extrapolation).
            Extrap : constant Float :=
              Float (Pos) / (P.Freq_Base ** (Float (2 * I) / Float (P.Dim)));
         begin
            if P.Yarn_On then
               --  Blend interpolation (Freq_Scale*extrap) and extrapolation by
               --  the correction-dim ramp: high-freq dims extrapolate, low-freq
               --  interpolate (llama.cpp rope_yarn).
               declare
                  Interp : constant Float := P.Freq_Scale * Extrap;
                  Y      : constant Float :=
                    (Float (I) - P.Corr_Low)
                    / Float'Max (0.001, P.Corr_High - P.Corr_Low);
                  Ramp   : constant Float :=
                    1.0 - Float'Min (1.0, Float'Max (0.0, Y));
               begin
                  Theta := Interp * (1.0 - Ramp) + Extrap * Ramp;
               end;
            else
               --  Freq_Scale = 1.0 default (no-op); < 1.0 = linear PI.
               Theta := Extrap * P.Freq_Scale;
            end if;
         end;
         --  Gemma full-attention layers scale wavelengths by rope_freqs
         --  (proportional / NTK RoPE): theta_i := theta_i / freq_factor_i.
         if P.Use_FF then
            Theta := Theta / Get_Flat (P.Freq_Factors, I + 1);
         end if;
         --  M_Scale = 1.0 unless YaRN set the attention temperature.
         Cos_Val := Cos (Theta) * P.M_Scale;
         Sin_Val := Sin (Theta) * P.M_Scale;

         if P.Interleaved then
            --  NORM / interleaved convention: pair adjacent dims (2i, 2i+1).
            --  llama.cpp permutes Llama Q/K weights for this layout.
            X1 := Get_Flat (X, 2 * I + 1);
            X2 := Get_Flat (X, 2 * I + 2);
            Set_Flat (Result, 2 * I + 1, X1 * Cos_Val - X2 * Sin_Val);
            Set_Flat (Result, 2 * I + 2, X2 * Cos_Val + X1 * Sin_Val);
         else
            X1 := Get_Flat (X, I + 1);              -- first half
            X2 := Get_Flat (X, I + Half_Dim + 1);   -- second half

            Set_Flat (Result, I + 1,            X1 * Cos_Val - X2 * Sin_Val);
            Set_Flat (Result, I + Half_Dim + 1, X2 * Cos_Val + X1 * Sin_Val);
         end if;
      end loop;

      return Result;
   end Apply;

end LLM_RoPE;
