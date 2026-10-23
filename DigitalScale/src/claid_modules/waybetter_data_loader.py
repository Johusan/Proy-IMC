from typing import Any, Dict
from claid.module.module import Module  # type: ignore
from data_types.digital_scale_pb2 import ImageArray, Image  # type: ignore
import cv2
import os
import pandas as pd
import sqlite3


class WayBetterDataLoader(Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, properties: Dict[str, Any]) -> None:
        print(f"Initializing! Got properties: {properties}")

        self.output_channel = self.publish("Images", ImageArray())

        limit = properties["limit"] if "limit" in properties else 2

        # Load images from path as specified in the config file.
        images = self.load_images_from_db(
            db_path=properties["db_path"],
            images_path=properties["images_path"],
            limit=limit,
        )
        self.output_channel.post(images)

    def load_images_from_db(self, db_path: str, images_path: str, limit: int) -> ImageArray:
        conn = sqlite3.connect(db_path)

        if limit > 0:
            weigh_ins_df = pd.read_sql_query(
                f"SELECT * FROM weigh_ins WHERE image_present = 1 LIMIT {limit};", conn
            )
        else:
            weigh_ins_df = pd.read_sql_query(
                "SELECT * FROM weigh_ins WHERE image_present = 1;", conn
            )

        # Load the images from the database
        image_array = ImageArray()

        for _, row in weigh_ins_df.iterrows():
            image_path = row["photo"]
            if not os.path.exists(os.path.join(images_path, image_path)):
                print(f"Image not found: {image_path}")
                continue
            image = cv2.imread(os.path.join(images_path, image_path))

            # Convert the image to bytes
            # Images can be png or jpg, so we need to check the file extension
            image_extension: str = os.path.splitext(image_path)[1]
            image_extension = image_extension.lower()

            if image_extension == ".png":
                image_bytes = cv2.imencode(".png", image)[1]
            elif image_extension == ".jpg":
                image_bytes = cv2.imencode(".jpg", image)[1]
            else:
                raise Exception(f"Unsupported image format: {image_extension}")

            # Create an Image message and add it to the ImageArray
            image_message = Image(
                width=image.shape[1], height=image.shape[0], data=image_bytes.tobytes()
            )
            image_array.images.append(image_message)

        return image_array
