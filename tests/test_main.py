"""main.py: o loop da câmera e os caminhos de erro, com câmera, detector e janela falsos."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as app_main
import synthetic_hand as sh
from core.hand_detector import HandObservation, ModelUnavailableError


class FakeCapture:
    def __init__(self, frames=1000, fail_after=None):
        self.frames, self.fail_after, self.reads, self.released = frames, fail_after, 0, False

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.full((480, 640, 3), 80, np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, pose="paz"):
        pts = sh.pose_points(pose)
        self.obs = HandObservation(pts, [(320 + 180 * x, 300 - 180 * y) for x, y, _ in pts])
        self.closed = False

    def detect(self, frame):
        return self.obs

    def close(self):
        self.closed = True


class MainLoopTestCase(unittest.TestCase):
    def run_main(self, argv=(), capture=None, detector=None, keys=None, detector_error=None, camera=True):
        capture = capture or FakeCapture()
        detector = detector or FakeDetector()
        keys = keys if keys is not None else [255] * 14 + [27]  # 14 quadros, depois ESC
        shown = []

        detector_factory = mock.Mock(return_value=detector)
        if detector_error:
            detector_factory.side_effect = detector_error

        patches = [
            mock.patch.object(app_main, "HandDetector", detector_factory),
            mock.patch.object(app_main, "open_camera", return_value=capture if camera else None),
            mock.patch.object(app_main.cv2, "imshow", side_effect=lambda w, f: shown.append(f)),
            mock.patch.object(app_main.cv2, "waitKey", side_effect=iter(keys)),
            mock.patch.object(app_main.cv2, "destroyAllWindows"),
            mock.patch.object(app_main, "window_closed", return_value=False),
            mock.patch.object(app_main.time, "sleep"),
        ]
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(out))
            code = app_main.main(list(argv))
        return code, out.getvalue(), capture, detector, shown

    def test_recognized_gesture_is_announced_once_and_exit_is_clean(self):
        code, out, capture, detector, shown = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(out.count("Gesto reconhecido: Sinal de Paz"), 1)
        self.assertEqual(len(shown), 15)
        self.assertTrue(capture.released, "a câmera deve ser liberada ao sair")
        self.assertTrue(detector.closed)

    def test_frames_shown_are_mirrored_like_a_mirror(self):
        # Marca a borda ESQUERDA do quadro da câmera; na tela ela deve aparecer à DIREITA.
        class MarkedCapture(FakeCapture):
            def read(self):
                self.reads += 1
                frame = np.full((480, 640, 3), 80, np.uint8)
                frame[240, 0] = (0, 0, 255)
                return True, frame

        _, _, _, _, shown = self.run_main(capture=MarkedCapture(), keys=[27])
        np.testing.assert_array_equal(shown[0][240, 639], (0, 0, 255))
        np.testing.assert_array_equal(shown[0][240, 0], (80, 80, 80))

    def test_q_key_also_quits(self):
        code, *_ = self.run_main(keys=[ord("q")])
        self.assertEqual(code, 0)

    def test_debug_and_landmark_keys_do_not_crash(self):
        code, *_ = self.run_main(keys=[ord("d"), ord("l"), ord("D"), 255, 27])
        self.assertEqual(code, 0)

    def test_temporary_read_failures_are_tolerated(self):
        # Antes: um único quadro ruim encerrava o programa.
        class Flaky(FakeCapture):
            def read(self):
                self.reads += 1
                if self.reads in (3, 4, 5):
                    return False, None
                return True, np.full((480, 640, 3), 80, np.uint8)

        code, _, capture, *_ = self.run_main(capture=Flaky(), keys=[255] * 6 + [27])
        self.assertEqual(code, 0)

    def test_camera_that_dies_ends_with_error_and_releases(self):
        code, out, capture, *_ = self.run_main(capture=FakeCapture(fail_after=2), keys=[255] * 5)
        self.assertEqual(code, 1)
        self.assertIn("parou de enviar", out)
        self.assertTrue(capture.released)

    def test_no_camera_gives_actionable_message(self):
        code, out, _, detector, _ = self.run_main(camera=False)
        self.assertEqual(code, 1)
        self.assertIn("--camera 1", out)
        self.assertTrue(detector.closed, "não deixar o detector aberto se a câmera falhou")

    def test_missing_model_gives_clean_error_instead_of_traceback(self):
        code, out, *_ = self.run_main(detector_error=ModelUnavailableError("Baixe manualmente: ..."))
        self.assertEqual(code, 1)
        self.assertIn("[ERRO]", out)

    def test_invalid_config_is_reported(self):
        path = os.path.join(tempfile.mkdtemp(), "ruim.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"name": "X", "pattern": [1, 2, 3], "image": "a.png"}], f)
        code, out, *_ = self.run_main(argv=["--config", path])
        self.assertEqual(code, 1)
        self.assertIn("Configuração de gestos inválida", out)

    def test_cli_defaults(self):
        args = app_main.parse_args([])
        self.assertEqual((args.camera, args.stability, args.release, args.debug), (0, 5, 4, False))


if __name__ == "__main__":
    unittest.main()
