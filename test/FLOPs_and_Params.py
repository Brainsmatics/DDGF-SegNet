import torch
from thop import profile
import unet_model as Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net.UNet(3, 2)
model.to(device)

input_img = torch.randn(1, 3, 512, 512).to(device)
flops, params = profile(model, inputs=(input_img,))

print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
print("Params=", str(params / 1e6) + '{}'.format("M"))

# FLOPs= 46.705410048G
# Params= 1.931033M
