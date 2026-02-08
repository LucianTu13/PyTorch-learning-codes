from PIL import Image
from torchvision import transforms

img = Image.open("test.jpg").convert("RGB")

transforms_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

img_tensor = transforms_pipeline(img)
print(f"处理后的形状: {img_tensor.shape}") # 输出: torch.Size([3, 224, 224])
print(f"像素值范围: min={img_tensor.min():.2f}, max={img_tensor.max():.2f}")