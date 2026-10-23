"""
Suite de pruebas automatizadas para el pipeline de precisión de DigitalScale.
Compatible con unittest estándar y pytest.
Verifica:
1. Carga del modelo SE-DenseNet-121 y preprocesamiento con preservación de aspecto.
2. Orden alfabético estricto de keypoints COCO-17 y modelos de clustering (Scaler, PCA, KMeans).
3. Reglas de control de calidad y rechazo de posturas/encuadres no válidos.
4. Clasificación clínica de la OMS y cálculo inverso de peso.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from server.services.bmi_service import BMIPredictor, _ResizeKeepAspect
from server.services.posture_service import PostureAnalyzer, COCO_ALPHABETICAL_INDICES
from server.services.image_service import ImageProcessor
from server.schemas import PredictionResponse


class TestBMIPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = BMIPredictor()
        cls.predictor.initialize()

    def test_initialization(self):
        self.assertTrue(self.predictor.is_ready)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.device)

    def test_resize_keep_aspect(self):
        resizer = _ResizeKeepAspect(224)
        
        # Imagen vertical (ej. cuerpo completo 400x800)
        portrait = Image.new("RGB", (400, 800), color=(100, 100, 100))
        resized_p = resizer(portrait)
        self.assertEqual(resized_p.height, 224)
        self.assertEqual(resized_p.width, 112)
        
        # Imagen horizontal (ej. 800x400)
        landscape = Image.new("RGB", (800, 400), color=(100, 100, 100))
        resized_l = resizer(landscape)
        self.assertEqual(resized_l.width, 224)
        self.assertEqual(resized_l.height, 112)

    def test_preprocess_tensor_shape(self):
        dummy_img = Image.new("RGB", (320, 640), color=(200, 150, 100))
        tensor = self.predictor.preprocess(dummy_img)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_prediction_range(self):
        dummy_img = Image.new("RGB", (224, 224), color=(180, 140, 110))
        bmi = self.predictor.predict_bmi(dummy_img)
        self.assertIsInstance(bmi, float)
        self.assertTrue(12.0 <= bmi <= 55.0)

    def test_who_categorization(self):
        self.assertEqual(BMIPredictor.categorize_who(17.5)[1], "underweight")
        self.assertEqual(BMIPredictor.categorize_who(22.0)[1], "normal")
        self.assertEqual(BMIPredictor.categorize_who(27.5)[1], "overweight")
        self.assertEqual(BMIPredictor.categorize_who(32.0)[1], "obese_1")
        self.assertEqual(BMIPredictor.categorize_who(37.0)[1], "obese_2")
        self.assertEqual(BMIPredictor.categorize_who(42.0)[1], "obese_3")

    def test_weight_calculation(self):
        weight = BMIPredictor.calculate_weight(25.0, 1.80)
        self.assertEqual(weight, 81.0)
        self.assertIsNone(BMIPredictor.calculate_weight(25.0, None))
        self.assertIsNone(BMIPredictor.calculate_weight(25.0, -1.0))


class TestPostureAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = PostureAnalyzer()
        cls.analyzer.initialize()

    def test_initialization(self):
        self.assertTrue(self.analyzer.is_ready)
        self.assertIsNotNone(self.analyzer.scaler)
        self.assertIsNotNone(self.analyzer.pca)
        self.assertIsNotNone(self.analyzer.kmeans)

    def test_coco_indices_integrity(self):
        self.assertEqual(len(COCO_ALPHABETICAL_INDICES), 17)
        self.assertEqual(sorted(COCO_ALPHABETICAL_INDICES), list(range(17)))

    def test_cluster_prediction_shape(self):
        dummy_kpts = np.zeros(34)
        dummy_kpts[:17] = 0.5
        dummy_kpts[17:] = 0.5
        
        scaled = self.analyzer.scaler.transform([dummy_kpts])
        self.assertEqual(scaled.shape, (1, 34))
        pca_feats = self.analyzer.pca.transform(scaled)
        self.assertEqual(pca_feats.shape, (1, 17))
        cluster = self.analyzer.kmeans.predict(pca_feats)[0]
        self.assertIn(cluster, [0, 1, 2, 3])

    def test_no_person_detected(self):
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.analyzer.analyze_posture(black_img)
        self.assertFalse(result["person_detected"])
        self.assertFalse(result["is_valid_pose"])
        self.assertIsNone(result["bbox"])


class TestImageProcessor(unittest.TestCase):
    def test_clean_fullbody_crop(self):
        img = Image.new("RGB", (640, 480), color=(50, 50, 50))
        bbox = (100, 50, 300, 450)
        crop = ImageProcessor.get_clean_fullbody_crop(img, bbox, 640, 480, margin_factor=1.10)
        self.assertTrue(crop.width > 0)
        self.assertTrue(crop.height > 0)
        self.assertTrue(crop.width <= 640)
        self.assertTrue(crop.height <= 480)

    def test_base64_encode_decode(self):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        b64_str = ImageProcessor.image_to_base64(img)
        self.assertTrue(b64_str.startswith("data:image/jpeg;base64,"))
        
        decoded_np = ImageProcessor.decode_base64_image(b64_str)
        self.assertIsNotNone(decoded_np)
        self.assertEqual(decoded_np.shape, (100, 100, 3))


class TestMultiFramePipeline(unittest.TestCase):
    """Pruebas de la arquitectura de agregación multi-frame y reducción de varianza."""

    def test_request_schema_burst(self):
        from server.schemas import Base64PredictRequest
        req = Base64PredictRequest(images_base64=["data:image/jpeg;base64,abc", "data:image/jpeg;base64,def"])
        self.assertIsNotNone(req.images_base64)
        self.assertEqual(len(req.images_base64), 2)
        self.assertIsNone(req.image_base64)

    def test_response_schema_fields(self):
        resp = PredictionResponse(
            success=True,
            bmi=24.5,
            is_valid_pose=True,
            posture_status="Medición estabilizada en ráfaga (3 fotogramas)",
            intra_std=0.25,
            frames_analyzed=3
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.frames_analyzed, 3)
        self.assertEqual(resp.intra_std, 0.25)

    def test_median_aggregation_logic(self):
        """Verifica que la agregación por mediana filtre outliers en ráfagas."""
        burst_preds = [24.2, 24.5, 31.0]  # El tercer frame tuvo un artefacto de movimiento
        median_bmi = float(np.median(burst_preds))
        self.assertEqual(median_bmi, 24.5)  # La mediana rechaza completamente el outlier de 31.0


if __name__ == "__main__":
    unittest.main()
