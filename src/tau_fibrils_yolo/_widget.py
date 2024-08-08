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
    QDoubleSpinBox,
)
from tau_fibrils_yolo.predict import FibrilsDetector
from tau_fibrils_yolo.measure import distance_measurement


class YoloDetectorWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.predictor = FibrilsDetector()

        self.shapes_layer = None
        self.filtered_shapes_layer = None
        self.crossover_shapes_layer = None
        self.filtered_lines_layer = None

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
        grid_layout.addWidget(self.btn, 1, 0, 1, 2)

        self.bx_probas = QDoubleSpinBox()
        self.bx_probas.setMinimum(0.0)
        self.bx_probas.setMaximum(1.0)
        self.bx_probas.setSingleStep(0.01)
        self.bx_probas.setValue(0.5)
        self.bx_probas.valueChanged.connect(self._filter_detections)
        grid_layout.addWidget(QLabel("Probability", self), 2, 0)
        grid_layout.addWidget(self.bx_probas, 2, 1)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 3, 0, 1, 2)

        self.cd_label = QLabel("CD: -", self)
        grid_layout.addWidget(self.cd_label, 4, 0, 1, 2)

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

        # Get the lines and crossover distances
        image = self.cb_image.currentData()
        distances = []
        line_data = []
        for box in boxes:
            distance, line_points = distance_measurement(box, image)
            distances.append(distance)
            line_data.append(line_points)

        
        lines_props = {
            "shape_type": 'line',
            "name": "Crossover distance",
            "opacity": 1.0, 
            "edge_width": 1,
            "edge_color": 'probability',
            "properties": {
                'probability': probas, 
                'distance': distances
            },
        }
        self.crossover_shapes_layer = self.viewer.add_shapes(line_data, **lines_props)

        shape_kwargs = {
            "shape_type": 'rectangle',
            "name":"Tau fibrils (Yolo)",
            "face_color": 'transparent',
            "opacity": 1.0,
            "edge_width": 1, 
            "edge_color": 'probability',
            "properties": {
                'probability': probas, 
                'distance': distances
            },
            # "text": {
            #     'string': '{distance:0.0f}', 
            #     'color': 'red', 
            #     'anchor': 'center'
            # }
        }

        self.shapes_layer = self.viewer.add_shapes(
            boxes,
            **shape_kwargs
        )
        
        self.pbar.setMaximum(1)

        self._filter_detections(self.bx_probas.value())

    def _filter_detections(self, probability_thresh):
        if self.shapes_layer is None:
            return
        
        shapes_data = np.array(self.shapes_layer.data)
        lines_data = np.array(self.crossover_shapes_layer.data)

        probabilities = self.shapes_layer.features['probability']
        distances = self.shapes_layer.features['distance']

        filtered_shapes = shapes_data[probabilities >= probability_thresh]
        filtered_lines = lines_data[probabilities >= probability_thresh]
        filtered_distances = distances[probabilities >= probability_thresh]
        # filtered_probabilities = probabilities[probabilities >= probability_thresh]

        if self.filtered_lines_layer is None:
            lines_props = {
                "shape_type": 'line',
                "name": "Crossover distance (filtered)",
                "opacity": 1.0, 
                "edge_width": 1,
                "edge_color": '#ff0000',
                # "properties": {
                #     'probability': filtered_probabilities, 
                #     'distance': filtered_distances
                # },
            }
            self.filtered_lines_layer = self.viewer.add_shapes(filtered_lines, **lines_props)

        if self.filtered_shapes_layer is None:
            shapes_props = {
                "shape_type": 'rectangle',
                "name": "Tau fibrils (filtered)",
                "face_color": 'transparent',
                "opacity": 1.0, 
                "edge_width": 1,
                "edge_color": '#ff0000',
                # "properties": {
                #     'probability': filtered_probabilities, 
                #     'distance': filtered_distances
                # },
                # "text": {
                #     'string': '{distance:0.0f}', 
                #     'color': 'red', 
                #     'anchor': 'center'
                # }
            }
            self.filtered_shapes_layer = self.viewer.add_shapes(filtered_shapes, **shapes_props)

        else:
            self.filtered_shapes_layer.data = filtered_shapes
            self.filtered_shapes_layer.edge_width = 1
            # self.filtered_shapes_layer.properties = {
            #     'probability': filtered_probabilities, 
            #     'distance': filtered_distances
            # }
            self.filtered_shapes_layer.refresh()

            self.filtered_lines_layer.data = filtered_lines
            self.filtered_lines_layer.edge_width = 1
            # self.filtered_lines_layer.properties = {
            #     'probability': filtered_probabilities, 
            #     'distance': filtered_distances
            # }
            self.filtered_lines_layer.refresh()

        # self.shapes_layer.visible = False
        # self.crossover_shapes_layer.visible = False

        self.cd_label.setText(f"CD: {np.mean(filtered_distances):.1f} ± {np.std(filtered_distances):.1f} px")
        # print(f"Mean crossover distance: ", np.mean(filtered_distances))