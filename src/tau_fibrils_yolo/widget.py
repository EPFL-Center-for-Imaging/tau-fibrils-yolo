import sys
from ultralytics import YOLO
import napari
import numpy as np
from skimage.io import imread
from skimage.exposure import rescale_intensity


MODEL_WEIGHTS_FILE = '/home/wittwer/code/tau-fibrils-yolo/runs/obb/train3/weights/best.pt'
IMGSZ = 640
MODEL = YOLO(
    model=MODEL_WEIGHTS_FILE,
    task='obb',
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

    print(f"Max: ", image.max())

    # Compute the pre-existing zero-padding at the top-left (if any)
    nx00 = (np.sum(image, axis=0) != 0).argmax()
    ny00 = (np.sum(image, axis=1) != 0).argmax()
    # image = image[ny0:, nx0:].copy()

    # Pad the image
    image_p = handle_padding(image, imgsz)

    # Convert the images to pseudo-RGB
    image_inputs = [to_rgb(c) for (c, (nx, ny)) in crop_generator(image_p, imgsz)]

    image_inputs_offset_coords_px = [
        (nx*imgsz, ny*imgsz) for (c, (nx, ny)) in crop_generator(image_p, imgsz)
    ]

    results = model.predict(
        source=image_inputs, 
        conf=0.25,  # Confidence threshold for detections.
        iou=0.5,  # Intersection over union threshold.
        imgsz=imgsz,  # Square resizing
        max_det=300,
        augment=False,
    )

    # First, remove the top-left delta XY (todo)
    first_crop = image_inputs[0][..., 0]
    nx0 = (np.sum(first_crop, axis=0) != 0).argmax()
    ny0 = (np.sum(first_crop, axis=1) != 0).argmax()

    all_coordinates = []
    for result, (nxp, nyp) in zip(results, image_inputs_offset_coords_px):
        boxes_coordinates = result.obb.xyxyxyxy.cpu().numpy()
        # probabilities = result.obb.conf.cpu().numpy()
        boxes_coordinates[..., 0] = boxes_coordinates[..., 0] + nyp - ny0
        boxes_coordinates[..., 1] = boxes_coordinates[..., 1] + nxp - nx0
        all_coordinates.append(boxes_coordinates)
    all_coordinates = np.vstack(all_coordinates)
    print(all_coordinates.shape)

    all_coordinates[..., 0] = all_coordinates[..., 0] + ny00
    all_coordinates[..., 1] = all_coordinates[..., 1] + nx00

    return all_coordinates


from magicgui import magicgui
from napari.types import ImageData, LayerDataTuple

@magicgui(auto_call=False)
def widget_yolo(image: ImageData) -> LayerDataTuple:

    print(image.shape)

    boxes_coordinates = handle_predict(image, MODEL, imgsz=IMGSZ)

    shape_kwargs = {
        "shape_type": 'rectangle',
        "edge_color": '#ff0000',
        "face_color": '#ff000044',
        "edge_width": 2, 
        "name":"Tau fibrils (Yolo)"
    }

    shapes_data = boxes_coordinates[:, :, ::-1]

    return (shapes_data, shape_kwargs, 'shapes')

if __name__=='__main__':
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(widget_yolo)
    napari.run()