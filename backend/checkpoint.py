import torch
ckpt = torch.load("best_mobilenetv3_plantvillage.pth", weights_only=True)
for k, v in ckpt.items():
    print(k, v.shape)