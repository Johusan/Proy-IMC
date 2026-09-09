# Filtering

The filtering uses three different methods to filter the data:

1. Bounding box confidence
2. Bounding box ratio
3. Posture cluster

(1) and (2) are based on the bounding box data.
(3) is based on body keypoint data.

## Bounding box and keypoint data

The following modules are used to get the bounding box and keypoint data:

- [bounding_box_detection.py](src/models/bounding_box_detection.py)
- [keypoint_detection.py](src/models/keypoint_detection.py)

These two modules are fully self-contained logic to detect the bounding box and keypoints in the image. Both have a corresponding utility script to run the forward pass on an image dataset and store the results to csv.

- [bounding_box_forwardpass.py](src/scripts/bounding_box_forwardpass.py)
- [keypoints_forwardpass.py](src/scripts/keypoints_forwardpass.py)


## Filtering module
The filtering logic needs all required data to be provided, i.e.:
1. Bounding box keypoints (provided by the keypoint detection module)
2. Bounding box confidence (provided by the bounding box detection module)
3. Posture cluster (see next section)

The filtering module is parameterized with thresholds for the bounding box confidence and ratio, and a list of posture clusters to exclude
- `bounding_box_confidence_threshold`
- `bounding_box_ratio_threshold`
- `outlying_posture_clusters`

The notebook [filtering.ipynb](report/simple_filtering/index.ipynb) shows how the class can be used. (Note, it uses some Myst visualiations for which you need to run a server, but you can still read the logic)

## Posture cluster data
The keypoints are clustered into posture clusters using the [keypoints.ipynb](report/simple_filtering/keypoints.ipynb) notebook.

However, these clusters are specifically made for the waybetter dataset. So I suggest that for the mobile app, it is easier to use just the bounding box-based filters (1) and (2) above.