from typing import Dict, Any, List
from claid.module import Module  # type: ignore
import pandas as pd
from src.helpers.pydantic_models import WaybetterImage



class PreprocessingModule(Module):
    def initialize(self, properties: Any) -> None:
        self.image_folder = properties.get("raw_images_folder", "")
        self.output_folder = properties.get("preprocessed_images_folder", "")

        print("Preprocessing module initialized")
        print(f"Image folder: {self.image_folder}")
        print(f"Output folder: {self.output_folder}")


class PersonDetectionFilter(Module):
    """Filters out images that do not contain a person."""

    def initialize(self, properties: Any) -> None:
        self.image_folder = properties.get("raw_images_folder", "")
        self.output_folder = properties.get("preprocessed_images_folder", "")
        self.confidence_threshold = properties.get("confidence_threshold", "")
        self.filter_report = 
        print("Person detection filter module initialized")
        print(f"Image folder: {self.image_folder}")
        print(f"Output folder: {self.output_folder}")

    def run(self, images: List[WaybetterImage]) -> None:
        # For each image, check the confidence of the person detection model
        for image in images:
            if image.bounding_box is None:
                # TODO: Deal with images that do not have bounding boxes
                continue

            if image.bounding_box.confidence > self.confidence_threshold:
                # Save the image to the output folder
                pass
            else:
                # Discard the image
                pass

class PersonRelativeSizeFilter(Module):
    """Filter out images where the bounding box is relatively small compared to the rest of the image"""
    