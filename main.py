import torchvision.models as models
import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np




model = models.mobilenet_v2(pretrained=True)
model.eval()

print(model.eval())