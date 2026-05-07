import torch
# from archs.lkdn_arch import LKDN as net
# from archs.GMIN_L_arch import GMIN_L as net
from archs.GLSKN_2_arch import GLSKN_2 as net
# from archs.CSINet_arch import CSINet as net
# from archs.lkdns_arch import LKDN_S as net
from thop import profile
# pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

# 将@ARCH_REGISTRY.register()注释
# model = net(
#     num_in_ch=3,
#     num_feat=48,
#     num_block=5,
#     num_out_ch=3,
#     upscale=4,
# )
model = net(
    inp_channels=3,
    out_channels=3,
    dim=42,
)
net_cls_str = f'{model.__class__.__name__}'


# thop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# input = torch.randn(1, 3, round(720/4), round(1280/4))
input = torch.randn(1, 3, round(256), round(256))
input=input.to('cuda')
flops, params = profile(model, inputs=(input,))
print("%s ------- params: %.4fMB ------- flops: %.4fG" % (net, params / (1000 ** 2), flops / (1000 ** 3)))

# inputs = torch.ones(1, 3, 320, 180, dtype=torch.float).to(device)
# flops, params = profile(model, (inputs, ))
# print(f'Network: {net_cls_str}, with flops(1280 x 720): {flops/1e9:.2f} GMac, with active parameters: {params/1e3} K.')
