import torch
import torchvision
import onnx
from lib.Network_Res2Net_GRA_NCD import Network
from torch.utils.mobile_optimizer import optimize_for_mobile
model = Network(imagenet_pretrained=False)
model.load_state_dict(torch.load("./Net_epoch_best.pth"))
model.eval()
example = torch.rand(1, 3, 640, 480)
model = torch.quantization.convert(model)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("SINet_model_ptl.ptl")
# model._save_for_lite_interpreter("SINet_model_with_pt.ptl")



