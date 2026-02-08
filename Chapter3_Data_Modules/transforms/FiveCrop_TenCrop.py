import torch
from PIL import Image
from torchvision import transforms

img = Image.open("test.jpg").convert("RGB")

five_crop_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.FiveCrop(224)
])

crops = five_crop_transforms(img)

to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

crops_tensors = torch.stack([to_tensor_norm(crop) for crop in crops])
print(f"FiveCrop 输出形状: {crops_tensors.shape}")