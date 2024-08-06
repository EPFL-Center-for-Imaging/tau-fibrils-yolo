import numpy as np
import napari
import napari.layers
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
    QSizePolicy,
)
from magicgui import magicgui
from tau_fibrils_yolo.predict import FibrilsDetector


@magicgui(auto_call=True)
def shapes_probability_threshold(shapes: napari.layers.Shapes, probability_thresh: float=0.5) -> napari.types.LayerDataTuple:
    shapes_data = np.array(shapes.data)
    probabilities = shapes.features['probability']

    filtered_shapes = shapes_data[probabilities >= probability_thresh]

    shapes_props = {
        "shape_type": 'rectangle',
        "name": "Filtered fibrils",
        "opacity": 1.0, 
        "face_color": 'transparent',
        "edge_width": 2,
        "edge_color": 'probability',
        "properties": {'probability': probabilities[probabilities >= probability_thresh]}
    }

    shapes.visible = False

    return (filtered_shapes, shapes_props, 'shapes')


class YoloDetectorWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.predictor = FibrilsDetector()

        # Layout
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        # Image
        self.cb_image = QComboBox()
        self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Image", self), 0, 0)
        grid_layout.addWidget(self.cb_image, 0, 1)
        
        # Compute button
        self.btn = QPushButton("Detect fibrils", self)
        self.btn.clicked.connect(self._start_detection)
        grid_layout.addWidget(self.btn, 3, 0, 1, 2)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 4, 0, 1, 2)

        # Setup layer callbacks
        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

    def _on_layer_change(self, e):
        self.cb_image.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Image):
                if x.data.ndim in [2, 3]:
                    self.cb_image.addItem(x.name, x.data)

    @thread_worker
    def _prediction_thread(self):
        return self.predictor.predict(self.selected_image)

    def _start_detection(self):
        self.selected_image = self.cb_image.currentData()
        if self.selected_image is None:
            return

        self.pbar.setMaximum(0)

        worker = self._prediction_thread()
        worker.returned.connect(self._load_in_viewer)
        worker.start()

    def _load_in_viewer(self, payload):
        """Callback from thread returning."""
        boxes, probas = payload

        if len(boxes) == 0:
            show_info("No fibrils were detected.")
            return

        boxes = boxes[:, :, ::-1]  # Correct X-Y coordinates for Napari.

        shape_kwargs = {
            "shape_type": 'rectangle',
            "edge_color": 'probability',
            "face_color": 'transparent',
            "edge_width": 2, 
            "name":"Tau fibrils (Yolo)",
            "opacity": 1.0,
            "properties": {'probability': probas}
        }

        self.viewer.add_shapes(
            boxes,
            **shape_kwargs
        )
        
        self.pbar.setMaximum(1)