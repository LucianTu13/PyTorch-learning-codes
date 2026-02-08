from PIL import Image

from torchvision import transforms


data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

img = Image.open("cat.jpg")
img_tensor = data_transform(img)