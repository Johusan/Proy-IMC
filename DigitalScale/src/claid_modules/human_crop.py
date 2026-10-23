import numpy as np
from PIL.Image import Image
from detectron2.engine import DefaultPredictor  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2 import model_zoo  # type: ignore


class CropToPerson(object):
    def __init__(self) -> None:
        # Initialize MediaPipe Pose model
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference

        self.predictor, self.cfg = DefaultPredictor(cfg), cfg

    def __call__(self, image: Image) -> Image:
        if not isinstance(image, Image):
            raise ValueError("Input must be a PIL Image.")

        # Convert PIL image to a NumPy array (as Detectron2 expects BGR format)
        image_np = np.array(image)[:, :, ::-1]  # Convert RGB (PIL) to BGR (OpenCV format)

        # Perform person detection
        outputs = self.predictor(image_np)

        # Extract the bounding box for the person class (class 0 in COCO dataset)
        instances = outputs["instances"]
        person_boxes = instances.pred_boxes[instances.pred_classes == 0]

        if len(person_boxes) > 0:
            # Assuming we want to crop around the first detected person
            box = person_boxes[0].tensor.cpu().numpy()[0].astype(int)
            x1, y1, x2, y2 = box

            # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))
            return cropped_image
        else:
            # If no person is detected, return the original image
            print("No person detected in the image. Returning the original image.")
            return image
