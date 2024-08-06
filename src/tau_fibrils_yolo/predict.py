import os
import numpy as np
from skimage.exposure import rescale_intensity
from ultralytics import YOLO
import pooch

MODEL_PATH = os.path.expanduser(
    os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".yolo")
)


def retreive_model():
    """Downloads the model weights from Zenodo."""
    print(" /!\ This model is on the Zenodo Sandbox (use it only for testing purposes) /!\ ")
    pooch.retrieve(
        url="https://sandbox.zenodo.org/records/99113/files/100ep.pt",
        known_hash="md5:2fc4be1e4feae93f75e335856be3083d",
        path=MODEL_PATH,
        progressbar=True,
        fname="yolo_fibrils_100ep.pt",
    )


def handle_padding(image, imgsz):
    rx, ry = image.shape
    nx = np.ceil(rx / imgsz).astype(int)
    ny = np.ceil(ry / imgsz).astype(int)

    pad_x = (nx * imgsz - rx) // 2
    pad_y = (ny * imgsz - ry) // 2

    image_p = np.zeros((nx*imgsz, ny*imgsz))
    image_p[pad_x:(pad_x+rx), pad_y:(pad_y+ry)] = image

    return image_p


def crop_generator(image_p, imgsz):
    rx, ry = image_p.shape
    nx = rx // imgsz
    ny = ry // imgsz

    for ix in range(nx):
        for iy in range(ny):
            crop = image_p[(ix*imgsz):(ix+1)*imgsz, (iy*imgsz):(iy+1)*imgsz]
            yield crop, (ix, iy)


def to_rgb(arr):
    return np.repeat(arr[..., None], repeats=3, axis=-1)


def handle_predict(image, model, imgsz):

    if len(image.shape) == 2:
        image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    elif len(image.shape) == 3:
        image = image[..., 0]

    # Compute the pre-existing zero-padding at the top-left (if any)
    nx00 = (np.sum(image, axis=0) != 0).argmax()
    ny00 = (np.sum(image, axis=1) != 0).argmax()

    # Pad the image
    image_p = handle_padding(image, imgsz)

    # Convert the images to pseudo-RGB
    image_inputs = [to_rgb(c) for (c, (nx, ny)) in crop_generator(image_p, imgsz)]

    image_inputs_offset_coords_px = [
        (nx*imgsz, ny*imgsz) for (c, (nx, ny)) in crop_generator(image_p, imgsz)
    ]

    results = model.predict(
        source=image_inputs, 
        conf=0.01,  # Confidence threshold for detections.
        iou=0.5,  # Intersection over union threshold.
        imgsz=imgsz,  # Square resizing
        max_det=500,  # Max detections per 640 x 640 crop
        augment=False,
    )

    # First, remove the top-left delta XY (todo)
    first_crop = image_inputs[0][..., 0]
    nx0 = (np.sum(first_crop, axis=0) != 0).argmax()
    ny0 = (np.sum(first_crop, axis=1) != 0).argmax()

    all_coordinates = []
    all_probabilities = []
    for result, (nxp, nyp) in zip(results, image_inputs_offset_coords_px):
        boxes_coordinates = result.obb.xyxyxyxy.cpu().numpy()
        boxes_coordinates[..., 0] = boxes_coordinates[..., 0] + nyp
        boxes_coordinates[..., 1] = boxes_coordinates[..., 1] + nxp
        all_coordinates.append(boxes_coordinates)

        probabilities = result.obb.conf.cpu().numpy()
        all_probabilities.extend(probabilities)

    all_coordinates = np.vstack(all_coordinates)
    all_probabilities = np.asarray(all_probabilities)

    all_coordinates[..., 0] = all_coordinates[..., 0] + ny00 - nx0
    all_coordinates[..., 1] = all_coordinates[..., 1] + nx00 - ny0

    return all_coordinates, all_probabilities


class FibrilsDetector():
    def __init__(self):
        retreive_model()
        self.model = YOLO(os.path.join(MODEL_PATH, "yolo_fibrils_100ep.pt"))
        self.imgsz = 640

    def predict(self, image: np.ndarray) -> np.ndarray:
        boxes, probas = handle_predict(image, self.model, self.imgsz)
        return boxes, probas