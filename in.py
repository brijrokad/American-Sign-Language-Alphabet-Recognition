import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import torch 
import torch.nn as nn
import pickle

model_ft = models.squeezenet1_1(pretrained=True)
num_ftrs = 128
model_ft.fc = nn.Linear(num_ftrs, 24)

use_gpu = torch.cuda.is_available()
use_gpu = False

if use_gpu:
    model_ft = model_ft.cuda()

checkpoint = torch.load('./tes_0.18823171658811008_0.9982788296041308.pth')
model_ft.load_state_dict(checkpoint)
model_ft.eval()

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(244),
   transforms.RandomSizedCrop(224),
   transforms.ToTensor(),
   normalize
])

# img_pil = Image.open('./hymenoptera_data/validation/bees/abeja.jpg')
# img_pil = Image.open('./hymenoptera_data/validation/ants/Hormiga.jpg')

img_pil = Image.open('./data/train/o/color_14_0399.png')
#img_pil = Image.open('./data/train/p/color_15_0399.png')
#img_pil = Image.open('./data/train/y/color_24_0399.png')
#img_pil = Image.open('./data/train/a/color_0_0399.png')

img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)

img_variable = Variable(img_tensor)
fc_out = model_ft(img_variable)
_, predicted = torch.max(fc_out.data, 1)

class_names = pickle.load(open('class_names.pkl', 'rb'))
print(class_names[predicted[0]])
print(class_names)

