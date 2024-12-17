import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms


ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'detaset size: {len(ds_train)}')

image, target = ds_train[1]
print(type(image), target)

plt.imshow(image, cmap='gray_r', vmin=0, vmax=255)
plt.title(target)
plt.show()

# PIL -> torch.Tensor
image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image,dtype=torch.float32, scale=True)
print(image.shape, image.dtype)
print(image.min(), image.max())