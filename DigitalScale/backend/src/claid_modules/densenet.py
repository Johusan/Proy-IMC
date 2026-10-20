import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader  # type: ignore
from densenet.Densenet import SEDensenet121  # type: ignore
from densenet.utils.train import Trainer  # type: ignore
from densenet.utils.OurDatasets import OurDatasets  # type: ignore
from typing import Any
from claid.module.module import Module  # type: ignore

from data_types.digital_scale_pb2 import ImageArray  # type: ignore


import os




class DenseNetInference(Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, properties: Any) -> None:
        self.input_channel = self.subscribe("InputImages", ImageArray(), self.on_data)

        self.model_checkpoint_path = properties["model_checkpoint_path"]

    def on_data(self, data: Any) -> None:
        images: ImageArray = data.get_data()
        print("Received images")
        os._exit(1)
        # self.run_inference(images)

    def run_inference(self, image_array: ImageArray) -> None:
        temp_folder = "temp_images"
        os.makedirs(temp_folder, exist_ok=True)
        for i, image in enumerate(image_array.images):
            # Write image to temporary file
            image_path = os.path.join(temp_folder, f"image_{i}.jpg")
            with open(image_path, "wb") as f:
                f.write(image.data)

        # Get Dataloader
        dataloader = get_dataloader(folder=temp_folder)

        forward_pass(self.model_checkpoint_path, dataloader)

        print("destroying all windows")
        os._exit(1)


def forward_pass(checkpoint_path: str, dataloader: torch.utils.data.DataLoader) -> None:
    # Load the model architecture
    model = SEDensenet121()

    # Move the model to the desired device
    DEVICE = torch.device("mps")
    model.to(DEVICE)

    # Load trained weights from the checkpoint if provided
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1,  # Meaningless learning rate, since we are only doing inference
        weight_decay=1,  # Meaningless weight decay value, since we are only doing inference
    )
    trainer = Trainer(model, DEVICE, optimizer, criterion)
    trainer.load(checkpoint_path)

    trainer.test(dataloader, sex="diff")

    return None


def get_dataloader(folder: str) -> torch.utils.data.DataLoader:
    test_dataset = OurDatasets(
        folder, "/", mode="3C", set="Our", kpts_fea=False, partition="test"
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=1
    )
    return test_loader
