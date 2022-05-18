import torch
import torchvision
from torchvision import transforms
from PIL import Image
from lib.Network_Res2Net_GRA_NCD import Network
path_ckpt = './Net_epoch_best.pth'
path_image = "./COD10K-CAM-1-Aquatic-3-Crab-71.jpg"
input_image = Image.open(path_image)
img_transform = transforms.Compose([
                    transforms.Resize((352, 352)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

input_tensor = img_transform(input_image).unsqueeze(0)
#input_tensor = torch.rand(1, 3, 640, 480)
print(input_tensor.shape)
path_ckpt = './Net_epoch_best.pth'
model = Network(imagenet_pretrained=False)
model.load_state_dict(torch.load(path_ckpt))
model.eval()
#model = torch.jit.load(path_ckpt)
input_names = [ "input" ]
output_names = [ "output1", "output2", "output3", "output4" ]
torch.onnx.export(model, input_tensor, "./Gaurav.onnx", input_names=input_names, output_names=output_names, opset_version=11)