# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine_torch import (
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)
from transformer_engine.pytorch import is_fp8_block_scaling_available, is_mxfp8_available
from transformer_engine.pytorch.optimizers.multi_tensor_apply import multi_tensor_applier


blockwise_available, reason_for_no_blockwise = is_fp8_block_scaling_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)


class TestBlockwisePartialCast:

    def _compute_partial_amax_reference(self, inp, amax, h, w, start_offset, block_len):
        n = inp.numel()
        full = torch.zeros(h * w, dtype=inp.dtype, device=inp.device)
        full[start_offset : start_offset + n].copy_(inp)
        full = full.view(h, w).abs()

        grid_h = (h + block_len - 1) // block_len
        grid_w = (w + block_len - 1) // block_len

        padded_full = torch.zeros(
            grid_h * block_len, grid_w * block_len, dtype=inp.dtype, device=inp.device
        )
        padded_full[:h, :w].copy_(full.view(h, w))

        padded_full_reshaped = padded_full.view(grid_h, block_len, grid_w, block_len)
        padded_full_permuted = padded_full_reshaped.permute(0, 2, 1, 3)

        amax_ref = padded_full_permuted.reshape(grid_h, grid_w, -1).max(dim=2).values
        amax.copy_(amax_ref)

    def _partial_cast_reference(self, inp, out, scale, h, w, start_offset, block_len, out_dtype):
        n = inp.numel()
        full = torch.empty(h * w, dtype=inp.dtype, device=inp.device)
        full[start_offset : start_offset + n].copy_(inp)
        full = full.float()

        # Expand scale to match full tensor size
        scale_expanded = scale.repeat_interleave(block_len, dim=0).repeat_interleave(block_len, dim=1)
        # Crop to actual size
        scale_expanded = scale_expanded[:h, :w]
        assert scale_expanded.dtype == torch.float32

        full_scaled = full.view(h, w) * scale_expanded

        tex_dtype_to_torch_dtype = {
            tex.DType.kFloat8E4M3: torch.float8_e4m3fn,
            tex.DType.kFloat8E5M2: torch.float8_e5m2,
        }
        full_out = full_scaled.to(tex_dtype_to_torch_dtype[out_dtype])

        full_out_flat = full_out.view(-1)
        out.copy_(full_out_flat[start_offset : start_offset + n].view(out.dtype))

    def _run_one_case(self, n, h, w, start_offset):
        full = torch.randn(h * w, dtype=torch.bfloat16, device="cuda")
        inp = full[start_offset : start_offset + n]
        block_len = 128

        grid_h = (h + block_len - 1) // block_len
        grid_w = (w + block_len - 1) // block_len

        # Partial amax cuda kernel
        amax = torch.zeros(grid_h, grid_w, dtype=torch.float32, device=inp.device)
        tex.fp8_block_scaling_compute_partial_amax(inp, amax, h, w, start_offset, block_len)
        # Partial amax pytorch reference
        amax_ref = torch.zeros_like(amax)
        self._compute_partial_amax_reference(inp, amax_ref, h, w, start_offset, block_len)
        # Check partial amax
        torch.testing.assert_close(amax, amax_ref, atol=0, rtol=0)

        # Compute full amax, scale and scale_inv
        full_amax = torch.zeros(grid_h, grid_w, dtype=torch.float32, device=inp.device)
        scale = torch.empty(grid_h, grid_w, dtype=torch.float32, device=inp.device)
        scale_inv = torch.empty(grid_h, grid_w, dtype=torch.float32, device=inp.device)
        self._compute_partial_amax_reference(full, full_amax, h, w, 0, block_len)
        multi_tensor_applier(
            multi_tensor_compute_scale_and_scale_inv,
            torch.zeros(1, dtype=torch.int, device=inp.device),
            [[amax], [scale], [scale_inv]],
            448.0,
            False,
            1e-5,
        )

        # Partial cast cuda kernel
        full_output = torch.empty(h * w, dtype=torch.uint8, device=inp.device).fill_(43)
        output = full_output[start_offset : start_offset + n]
        tex.fp8_block_scaling_partial_cast(
            inp,
            output,
            scale,
            h,
            w,
            start_offset,
            block_len,
            tex.DType.kFloat8E4M3,
        )
        # Partial cast pytorch reference
        full_output_ref = torch.empty(h * w, dtype=torch.uint8, device=inp.device).fill_(43)
        output_ref = full_output_ref[start_offset : start_offset + n]
        self._partial_cast_reference(
            inp,
            output_ref,
            scale,
            h,
            w,
            start_offset,
            block_len,
            tex.DType.kFloat8E4M3,
        )

        # Check partial cast results
        print(inp)
        print(scale)
        print(output)
        print(output_ref)
        print(full_output)
        print(full_output_ref)
        print(full_output.view(h, w)[0])
        print(full_output_ref.view(h, w)[0])
        torch.testing.assert_close(output, output_ref, atol=0, rtol=0)
        torch.testing.assert_close(full_output, full_output_ref, atol=0, rtol=0)

    @pytest.mark.skipif(not blockwise_available, reason=reason_for_no_blockwise)
    def test_fp8_block_scaling_partial_cast(self):
        torch.cuda.manual_seed(1234)

        self._run_one_case(3, 32, 64, 32)
        # self._run_one_case(64 * 64 - 2, 64, 64, 1)
        # self._run_one_case(16384 * 6144, 16384, 6144, 0)
        # self._run_one_case(32768, 256, 128, 0)
        # self._run_one_case(131072, 768, 256, 0)
        # self._run_one_case(65536, 768, 256, 131072)
        # self._run_one_case(98304, 128, 768, 0)


class TestMXFP8PartialCast:

    def _compute_partial_amax_reference(self, inp, amax_rowwise, amax_colwise, h, w, start_offset):
        n = inp.view(-1).size(0)
        if n == h * w:
            full = inp.view(-1)
        else:
            full = torch.zeros(h * w, dtype=inp.dtype, device=inp.device)
            full[start_offset : start_offset + n].copy_(inp)
        full = torch.abs(full)
        _amax_rowwise, _ = torch.max(full.view(h, w // 32, 32), dim=2)
        amax_rowwise[:h, : (w // 32)].copy_(_amax_rowwise)
        _amax_colwise, _ = torch.max(full.view(h // 32, 32, w), dim=1)
        amax_colwise[: (h // 32), :w].copy_(_amax_colwise)

    def _partial_cast_reference(
        self,
        inp,
        rowwise_out,
        colwise_out,
        rowwise_inv_scale,
        colwise_inv_scale,
        h,
        w,
        start_offset,
    ):
        rowwise_scale = ((254 - rowwise_inv_scale.int()) * 2**23).view(torch.float32)
        colwise_scale = ((254 - colwise_inv_scale.int()) * 2**23).view(torch.float32)
        n = inp.view(-1).size(0)
        if n == h * w:
            full = inp
        else:
            full = torch.empty(h * w, dtype=inp.dtype, device=inp.device)
            full[start_offset : start_offset + n].copy_(inp)
        full = full.float()
        rowwise_scale = rowwise_scale[:h, : (w // 32)].contiguous().float()
        colwise_scale = colwise_scale[: (h // 32), :w].contiguous().float()
        scaled = (full.view(-1, 32) * rowwise_scale.view(-1, 1)).view(-1)
        rowwise_out.copy_(
            scaled[start_offset : start_offset + n].to(torch.float8_e4m3fn).view(rowwise_out.dtype)
        )
        scaled = (full.view(h // 32, 32, w) * colwise_scale.view(h // 32, 1, w)).view(-1)
        colwise_out.copy_(
            scaled[start_offset : start_offset + n].to(torch.float8_e4m3fn).view(colwise_out.dtype)
        )

    def _run_one_case(self, n, h, w, start_offset):
        inp = torch.randn(n, dtype=torch.bfloat16, device="cuda")

        rowwise_padding = [128, 4]
        colwise_padding = [4, 128]

        def _pad(x, padding):
            return (x + padding - 1) // padding * padding

        rowwise_shape = [_pad(h, rowwise_padding[0]), _pad(w // 32, rowwise_padding[1])]
        colwise_shape = [_pad(h // 32, colwise_padding[0]), _pad(w, colwise_padding[1])]

        # Partial amax cuda kernel
        amax_rowwise = torch.zeros(*rowwise_shape, dtype=inp.dtype, device=inp.device)
        amax_colwise = torch.zeros(*colwise_shape, dtype=inp.dtype, device=inp.device)
        tex.mxfp8_scaling_compute_partial_amax(inp, amax_rowwise, amax_colwise, h, w, start_offset)

        # Partial amax pytorch reference
        amax_rowwise_ref = torch.zeros(*rowwise_shape, dtype=inp.dtype, device=inp.device)
        amax_colwise_ref = torch.zeros(*colwise_shape, dtype=inp.dtype, device=inp.device)
        self._compute_partial_amax_reference(
            inp, amax_rowwise_ref, amax_colwise_ref, h, w, start_offset
        )

        # Check partial amax
        torch.testing.assert_close(amax_rowwise, amax_rowwise_ref, atol=0, rtol=0)
        torch.testing.assert_close(amax_colwise, amax_colwise_ref, atol=0, rtol=0)

        # Calculate scales and scale_invs
        scale_inv_rowwise = torch.empty_like(amax_rowwise).to(torch.uint8)
        scale_inv_colwise = torch.empty_like(amax_colwise).to(torch.uint8)
        multi_tensor_applier(
            multi_tensor_compute_scale_inv_e8m0,
            None,
            [
                [amax_rowwise, amax_colwise],
                [scale_inv_rowwise, scale_inv_colwise],
            ],
        )

        # Partial cast cuda kernel
        output_rowwise = torch.empty_like(inp).to(torch.uint8)
        output_colwise = torch.empty_like(inp).to(torch.uint8)
        tex.mxfp8_scaling_partial_cast(
            inp,
            output_rowwise,
            output_colwise,
            scale_inv_rowwise,
            scale_inv_colwise,
            h,
            w,
            start_offset,
        )

        # Partial cast pytorch reference
        output_rowwise_ref = torch.empty_like(inp).to(torch.uint8)
        output_colwise_ref = torch.empty_like(inp).to(torch.uint8)
        self._partial_cast_reference(
            inp,
            output_rowwise_ref,
            output_colwise_ref,
            scale_inv_rowwise,
            scale_inv_colwise,
            h,
            w,
            start_offset,
        )

        # Check partial cast results
        torch.testing.assert_close(output_rowwise, output_rowwise_ref, atol=0, rtol=0)
        torch.testing.assert_close(output_colwise, output_colwise_ref, atol=0, rtol=0)

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_mxfp8_scaling_partial_cast(self):
        torch.cuda.manual_seed(1234)

        self._run_one_case(3, 32, 64, 31)
        self._run_one_case(64 * 64 - 2, 64, 64, 1)
        self._run_one_case(16384 * 6144, 16384, 6144, 0)
        self._run_one_case(32768, 256, 128, 0)
        self._run_one_case(131072, 768, 256, 0)
        self._run_one_case(65536, 768, 256, 131072)
        self._run_one_case(98304, 128, 768, 0)
