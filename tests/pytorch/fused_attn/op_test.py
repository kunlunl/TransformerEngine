import math
import torch
import transformer_engine
import transformer_engine_extensions as tex
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward


max_seqlen = 16*1024//4
batch = 8
num_heads = 12
head_dim = 128

cu_seqlens = [0]
for i in range(batch):
    cu_seqlens.append(cu_seqlens[i] + max_seqlen)
cu_seqlens = torch.Tensor(cu_seqlens).to(torch.int32).cuda()
print("cu_seqlens:", cu_seqlens)
total_len = batch * max_seqlen

seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
print("seqlens:", seqlens)

cu_lens = cu_seqlens // 2
print("cu_lens:", cu_lens)

out_offset_0 = cu_seqlens
out_offset_1 = cu_seqlens[:-1] + seqlens // 2
print("out_offset_0:", out_offset_0)
print("out_offset_1:", out_offset_1)

lse_offset_0 = torch.Tensor([0] * (batch+1)).to(torch.int32).cuda()
lse_offset_1 = seqlens // 2
print("lse_offset_0:", lse_offset_0)
print("lse_offset_1:", lse_offset_1)

'''
lse1 = torch.rand(batch, num_heads, max_seqlen).to(torch.double).cuda()
lse2 = torch.rand(batch, num_heads, max_seqlen//2).to(torch.float).cuda()
lse3 = lse1.clone()
tex.thd_lse_correction(lse1, lse2, cu_seqlens, total_len)
tex.thd_seg_lse_correction(lse3, lse2, lse_offset_1, lse_offset_0, cu_lens, total_len//2)
print('err Duration:', (lse3 - lse1).abs().sum())
'''

'''
q1 = torch.randn(total_len//2, num_heads, head_dim).cuda()
q2 = torch.randn(total_len, num_heads, head_dim).cuda()
q3 = tex.thd_read_half_tensor(q2, cu_seqlens, 1)
tex.thd_segment_copy(q1, q2, cu_lens, out_offset_1, cu_lens, total_len//2)
print('err Duration:', (q1 - q3).abs().sum())
'''

#'''
out1 = torch.randn(total_len, num_heads, head_dim).cuda()
out2 = torch.randn(total_len//2, num_heads, head_dim).cuda()
out3 = out1.clone()
lse1 = torch.rand(batch, num_heads, max_seqlen).to(torch.float).cuda()
lse2 = torch.rand(batch, num_heads, max_seqlen//2).to(torch.float).cuda()
tex.thd_out_correction(out1, out2, lse1, lse2, cu_seqlens, True)
#tex.thd_seg_out_correction(out3, out2, lse1, lse2,
#                           out_offset_1, cu_lens, lse_offset_1, lse_offset_0,
#                           cu_lens, total_len//2)
print('err Duration:', (out3 - out1).abs().sum())
#'''
