import torch
import torchvision
import onnx
from lib.Network_Res2Net_GRA_NCD import Network
model = Network(imagenet_pretrained=False)
model.load_state_dict(torch.load("./Net_epoch_best.pth"))
model.eval()
example = torch.rand(1, 3, 640, 480)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./SINet-model.pt")





