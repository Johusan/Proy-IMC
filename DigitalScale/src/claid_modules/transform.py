import os
from numpy import ndarray
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore
import torchvision.transforms.functional as F  # type: ignore
import cv2
from typing import TypedDict, Literal, Dict, Any
from PIL.Image import Image

from claid.module import Module  # type: ignore
from datetime import timedelta

from claid_modules.human_crop import CropToPerson


class DatasetItem(TypedDict):
    original_image_path: str
    original_image: ndarray
    current_image: ndarray


class ImageTransformDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        img_size: int,
        transform_strategy: Literal["center_crop", "human_crop", "no_crop"],
    ):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.image_files = os.listdir(image_dir)

        if transform_strategy == "center_crop":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(self.img_size),
                    transforms.Pad(self.img_size),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                ]
            )
        elif transform_strategy == "human_crop":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    CropToPerson(),
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                ]
            )
        elif transform_strategy == "no_crop":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),
                    # transforms.Pad(self.img_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> DatasetItem:
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = cv2.imread(img_path, flags=1)[:, :, ::-1]

        processed_image = self.transform(img)

        return {
            "original_image_path": img_path,
            "original_image": img,
            "current_image": processed_image,
        }

    def save_transformed_images(self) -> None:
        """
        Save the transformed images to the output directory
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for idx in range(len(self)):
            item = self[idx]
            image, filename = item["current_image"], os.path.basename(
                item["original_image_path"]
            )
            # Convert the tensor back to a PIL image
            pil_image: Image = F.to_pil_image(image)
            # Save the image to the output directory
            output_path = os.path.join(self.output_dir, filename)
            pil_image.save(output_path)


class TransformationModule(Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def initialize(self, properties: Dict[str, Any]) -> None:
        self.image_dir = properties.get("image_dir", "")
        self.output_dir = properties.get("output_dir", "")
        self.img_size = properties.get("img_size", 224)
        print("Transformation module initialized")
        self.run()
        self.register_periodic_function(
            "Test", self.periodic_function, timedelta(milliseconds=1000)
        )

    def run(self) -> None:
        dataset = ImageTransformDataset(
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            img_size=self.img_size,
            transform_strategy="center_crop",
        )
        dataset.save_transformed_images()

    def periodic_function(self) -> None:
        print("PeriodsdfsfdicFunction")
