from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.config import instanciate_module


class MedMNISTDataset(Dataset):
    """
    PyTorch Dataset wrapper for MedMNIST datasets using dynamic instantiation.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = 'BreastMNIST',
        image_size: int = 128,
        split: str = 'train',
        transform=None
    ):
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        self.dataset = instanciate_module(
            module_name='medmnist',
            class_name=dataset_name,
            params={
                "root": data_dir,
                "download": True,
                "transform": self.transform,
                "size": image_size,
                "split": split,
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        return x, y.squeeze(0)

# class KSpaceBreastMNIST(BaseMedMnistDataset):
#     def __init__(self, data_dir, image_size: int = 128, split='train', transform=None):
#         super().__init__(data_dir, image_size, split, transform)

#     def __getitem__(self, idx):
#         image, label = self.b_mnist[idx]
#         image_np = image.squeeze().numpy()
#         kspace_image = np.fft.fftshift(np.fft.fft2(image_np))
#         kspace_abs = np.abs(kspace_image)
#         kspace_tensor = cv2.resize(
#             kspace_abs, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
#         kspace_tensor = torch.tensor(kspace_tensor, dtype=torch.float32)
#         kspace_tensor = kspace_tensor.unsqueeze(0)
#         label = label.squeeze(0).astype('float32')

#         return kspace_tensor, label
