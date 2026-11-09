"""Teste de ponta a ponta, sem webcam: mão sintética -> dedos -> gesto -> imagem na tela.

Usa o patterns.json e as imagens reais do projeto; só a câmera e o MediaPipe são
substituídos por um detector falso que "enxerga" uma mão sintética.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synthetic_hand as sh
from core.app import GestureApp, ImageCache
from core.hand_detector import HandObservation
from core.pattern_matcher import PatternMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, BG = 640, 480, 80


def observation(pose, **transform):
    pts = sh.pose_points(pose, **transform)
    image_points = [(320 + 180 * x, 300 - 180 * y) for x, y, _ in pts]  # projeção ortográfica simples
    return HandObservation(points=pts, image_points=image_points)


class FakeDetector:
    """Devolve a observação que o teste definiu (ou None = nenhuma mão)."""

    def __init__(self):
        self.current = None

    def detect(self, frame):
        return self.current


def card_region():
    """Retângulo do canto superior direito onde a imagem do gesto é desenhada."""
    margin = max(10, H // 45)
    box = int(min(H * 0.46, W * 0.38))
    return slice(margin, margin + box), slice(W - box - margin, W - margin)


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.warnings = []
        self.detector = FakeDetector()
        self.app = self.make_app(os.path.join(BASE_DIR, "patterns.json"), BASE_DIR)

    def make_app(self, config, base_dir):
        return GestureApp(
            self.detector,
            PatternMatcher(config, stability_frames=5, release_frames=4),
            ImageCache(base_dir, warn=self.warnings.append),
            show_landmarks=True,
        )

    def run_frames(self, obs, n):
        self.detector.current = obs
        result = None
        for _ in range(n):
            frame = np.full((H, W, 3), BG, np.uint8)
            result = self.app.process(frame)
        return result

    def test_gesture_shows_its_image_on_screen(self):
        result = self.run_frames(observation("paz"), 14)
        self.assertEqual(result.entry["name"], "Sinal de Paz")
        ys, xs = card_region()
        self.assertFalse(np.all(result.frame[ys, xs] == BG), "a imagem do gesto não apareceu na tela")

    def test_image_is_not_shown_before_the_gesture_is_stable(self):
        result = self.run_frames(observation("paz"), 3)
        self.assertIsNone(result.entry)
        ys, xs = card_region()
        self.assertTrue(np.all(result.frame[ys, xs] == BG))

    def test_image_fades_in(self):
        ys, xs = card_region()
        first = self.run_frames(observation("mao_aberta"), 5)  # acabou de confirmar: opacidade baixa
        full = self.run_frames(observation("mao_aberta"), 10)
        diff_first = np.abs(first.frame[ys, xs].astype(int) - BG).sum()
        diff_full = np.abs(full.frame[ys, xs].astype(int) - BG).sum()
        self.assertGreater(diff_full, diff_first)

    def test_image_disappears_when_hand_leaves(self):
        self.run_frames(observation("punho"), 14)
        result = self.run_frames(None, 5)
        self.assertIsNone(result.entry)
        ys, xs = card_region()
        self.assertTrue(np.all(result.frame[ys, xs] == BG), "imagem ficou presa na tela")

    def test_changed_flag_fires_once_per_new_gesture(self):
        self.detector.current = observation("paz")
        changes = []
        for _ in range(20):
            r = self.app.process(np.full((H, W, 3), BG, np.uint8))
            if r.changed and r.entry:
                changes.append(r.entry["name"])
        self.assertEqual(changes, ["Sinal de Paz"])

    def test_sequence_of_gestures_is_recognized_in_order(self):
        seen = []
        for pose in ("punho", "indicador", "paz", "mao_aberta", "positivo"):
            r = self.run_frames(observation(pose), 12)
            seen.append(r.entry["name"])
        self.assertEqual(
            seen,
            ["Punho Fechado", "Indicador Levantado", "Sinal de Paz", "Mão Aberta", "Sinal de Positivo"],
        )

    def test_recognition_works_with_rotated_and_tilted_hand(self):
        for rz in (0, 90, 180, -120):
            with self.subTest(rotation=rz):
                self.app = self.make_app(os.path.join(BASE_DIR, "patterns.json"), BASE_DIR)
                r = self.run_frames(observation("paz", rx=25, ry=-20, rz=rz), 12)
                self.assertEqual(r.entry["name"], "Sinal de Paz")

    def test_thumb_does_not_matter_for_wildcard_gestures(self):
        for pose in ("punho", "punho_polegar_lateral"):
            self.app = self.make_app(os.path.join(BASE_DIR, "patterns.json"), BASE_DIR)
            self.assertEqual(self.run_frames(observation(pose), 12).entry["name"], "Punho Fechado")

    def test_missing_image_does_not_crash_and_warns_once(self):
        folder = tempfile.mkdtemp()
        config = os.path.join(folder, "patterns.json")
        with open(config, "w", encoding="utf-8") as f:
            json.dump([{"name": "Sinal de Paz", "pattern": [None, 1, 1, 0, 0], "image": "assets/sumiu.png"}], f)
        self.app = self.make_app(config, folder)

        result = self.run_frames(observation("paz"), 20)
        self.assertEqual(result.entry["name"], "Sinal de Paz")  # o reconhecimento continua funcionando
        self.assertEqual(len(self.warnings), 1, "deveria avisar uma única vez, não a cada quadro")
        self.assertIn("sumiu.png", self.warnings[0])

    def test_debug_mode_and_landmark_toggle_do_not_crash(self):
        self.app.debug = True
        self.run_frames(observation("paz"), 8)
        self.assertFalse(self.app.toggle_landmarks())
        self.run_frames(observation("paz"), 2)
        self.assertTrue(self.app.toggle_landmarks())
        self.assertFalse(self.app.toggle_debug())

    def test_works_on_small_and_large_frames(self):
        for w, h in ((320, 240), (1280, 720), (1920, 1080)):
            with self.subTest(size=(w, h)):
                self.app = self.make_app(os.path.join(BASE_DIR, "patterns.json"), BASE_DIR)
                self.detector.current = observation("mao_aberta")
                for _ in range(12):
                    r = self.app.process(np.full((h, w, 3), BG, np.uint8))
                self.assertEqual(r.entry["name"], "Mão Aberta")
                self.assertEqual(r.frame.shape, (h, w, 3))


if __name__ == "__main__":
    unittest.main()
