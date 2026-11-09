"""HandDetector: conversão de resultados, download do modelo e compatibilidade com o MediaPipe."""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_detector import ModelUnavailableError, ensure_model, observation_from_result

HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None


def fake_landmarks(offset=0.0):
    return [SimpleNamespace(x=0.1 + i * 0.01 + offset, y=0.2 + i * 0.01, z=0.0) for i in range(21)]


class ObservationFromResultTestCase(unittest.TestCase):
    def test_no_hands_returns_none(self):
        self.assertIsNone(observation_from_result(SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[]), 640, 480))

    def test_image_points_are_scaled_to_pixels(self):
        result = SimpleNamespace(hand_landmarks=[fake_landmarks()], hand_world_landmarks=[])
        obs = observation_from_result(result, 640, 480)
        self.assertEqual(len(obs.image_points), 21)
        self.assertAlmostEqual(obs.image_points[0][0], 0.1 * 640)
        self.assertAlmostEqual(obs.image_points[0][1], 0.2 * 480)

    def test_world_landmarks_are_preferred_for_geometry(self):
        world = [SimpleNamespace(x=float(i), y=0.0, z=0.0) for i in range(21)]
        result = SimpleNamespace(hand_landmarks=[fake_landmarks()], hand_world_landmarks=[world])
        obs = observation_from_result(result, 640, 480)
        self.assertEqual(obs.points[5], (5.0, 0.0, 0.0))

    def test_fallback_scales_x_y_and_z_the_same_way(self):
        # Sem world landmarks, x e y viram pixels; z precisa seguir a escala de x,
        # senão ângulos e distâncias 3D ficam distorcidos pela proporção da imagem.
        lm = [SimpleNamespace(x=0.5, y=0.5, z=0.1)] * 21
        obs = observation_from_result(SimpleNamespace(hand_landmarks=[lm], hand_world_landmarks=[]), 640, 480)
        self.assertEqual(obs.points[0], (320.0, 240.0, 64.0))

    def test_truncated_landmark_list_returns_none(self):
        result = SimpleNamespace(hand_landmarks=[fake_landmarks()[:10]], hand_world_landmarks=[])
        self.assertIsNone(observation_from_result(result, 640, 480))


class EnsureModelTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "models", "hand_landmarker.task")
        silence = contextlib.redirect_stdout(io.StringIO())  # ensure_model imprime o progresso
        silence.__enter__()
        self.addCleanup(silence.__exit__, None, None, None)

    @staticmethod
    def good_download(url, dest):
        with open(dest, "wb") as f:
            f.write(b"PK" + b"\0" * 1_200_000)

    def test_existing_valid_model_is_not_downloaded_again(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "wb") as f:
            f.write(b"PK" + b"\0" * 1_200_000)

        def must_not_run(url, dest):
            raise AssertionError("não deveria baixar de novo")

        self.assertEqual(ensure_model(self.path, downloader=must_not_run), self.path)

    def test_downloads_missing_model(self):
        self.assertEqual(ensure_model(self.path, downloader=self.good_download), self.path)
        self.assertTrue(os.path.isfile(self.path))
        self.assertFalse(os.path.exists(self.path + ".part"))

    def test_download_failure_gives_actionable_error_and_leaves_no_partial_file(self):
        def offline(url, dest):
            with open(dest, "wb") as f:
                f.write(b"PK")
            raise OSError("sem internet")

        with self.assertRaises(ModelUnavailableError) as ctx:
            ensure_model(self.path, downloader=offline)
        self.assertIn("Baixe manualmente", str(ctx.exception))
        self.assertIn(self.path, str(ctx.exception))
        self.assertFalse(os.path.exists(self.path + ".part"))
        self.assertFalse(os.path.exists(self.path))

    def test_binary_model_without_zip_header_is_accepted(self):
        # Não assumimos o formato interno do .task; quem valida é o MediaPipe.
        def binary(url, dest):
            with open(dest, "wb") as f:
                f.write(b"\x1c\x00\x00\x00TFL3" + b"\x01" * 1_200_000)

        self.assertEqual(ensure_model(self.path, downloader=binary), self.path)

    def test_html_error_page_is_rejected_as_model(self):
        def html_page(url, dest):
            with open(dest, "wb") as f:
                f.write(b"<html>" + b"x" * 1_200_000)

        with self.assertRaises(ModelUnavailableError):
            ensure_model(self.path, downloader=html_page)
        self.assertFalse(os.path.exists(self.path))


@unittest.skipUnless(HAS_MEDIAPIPE, "mediapipe não instalado")
class MediaPipeCompatibilityTestCase(unittest.TestCase):
    """Garante que a API que o projeto usa existe na versão instalada.

    O código original usava `mp.solutions`, removido nas versões novas, e só
    quebrava quando o programa abria a câmera. Este teste pega isso cedo.
    """

    def test_tasks_api_symbols_exist(self):
        from mediapipe.tasks.python import BaseOptions, vision

        self.assertTrue(hasattr(vision, "HandLandmarker"))
        self.assertTrue(hasattr(vision, "HandLandmarkerOptions"))
        self.assertTrue(hasattr(vision.RunningMode, "VIDEO"))
        self.assertTrue(hasattr(vision.HandLandmarker, "detect_for_video"))
        self.assertIn("model_asset_buffer", BaseOptions.__dataclass_fields__)

    def test_options_we_pass_are_accepted(self):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=b"x"),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.assertEqual(options.num_hands, 1)

    def test_image_wrapper_accepts_our_frames(self):
        import mediapipe as mp
        import numpy as np

        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.assertEqual((image.width, image.height), (64, 48))

    def test_detector_without_model_fails_with_clear_error(self):
        from core.hand_detector import HandDetector

        missing = os.path.join(tempfile.mkdtemp(), "nao_existe", "m.task")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ModelUnavailableError):
            HandDetector(model_path=missing)


if __name__ == "__main__":
    unittest.main()
