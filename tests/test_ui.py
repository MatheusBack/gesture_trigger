"""Texto com acentos, esqueleto da mão e painel de debug."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synthetic_hand as sh
from core.ui import HAND_CONNECTIONS, TextRenderer, draw_debug_panel, draw_hand


class TextRendererTestCase(unittest.TestCase):
    def test_accented_text_renders_visible_pixels(self):
        bgra = TextRenderer().render("Mão Aberta · ção", 24)
        self.assertEqual(bgra.shape[2], 4)
        self.assertGreater(int((bgra[:, :, 3] > 0).sum()), 200)

    def test_same_text_is_served_from_cache(self):
        t = TextRenderer()
        self.assertIs(t.render("Olá", 20), t.render("Olá", 20))

    def test_draw_changes_the_frame_and_right_align_stays_inside(self):
        frame = np.zeros((120, 300, 3), np.uint8)
        TextRenderer().draw(frame, "30 FPS", 290, 10, 20, align="right")
        self.assertGreater(int(frame.sum()), 0)
        self.assertEqual(int(frame[:, 290:].sum()), 0, "texto alinhado à direita não pode passar de x=290")

    def test_drawing_text_off_screen_does_not_crash(self):
        frame = np.zeros((50, 50, 3), np.uint8)
        TextRenderer().draw(frame, "texto longo demais para caber na tela", 30, 40, 30)


class HandDrawingTestCase(unittest.TestCase):
    def test_hand_has_21_points_and_connections_are_valid(self):
        self.assertTrue(all(0 <= a < 21 and 0 <= b < 21 for a, b in HAND_CONNECTIONS))

    def test_draw_hand_colors_fingers_by_state(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        pts = [(320 + 200 * x, 300 - 200 * y) for x, y, _ in sh.pose_points("paz")]
        draw_hand(frame, pts, [0, 1, 1, 0, 0])
        self.assertGreater(int(frame.sum()), 0)
        green = int(((frame[:, :, 1] > 200) & (frame[:, :, 0] < 150)).sum())
        orange = int(((frame[:, :, 2] > 200) & (frame[:, :, 0] < 100) & (frame[:, :, 1] < 150)).sum())
        self.assertGreater(green, 0, "dedos esticados deveriam aparecer verdes")
        self.assertGreater(orange, 0, "dedos dobrados deveriam aparecer laranja")

    def test_debug_panel_draws(self):
        frame = np.zeros((300, 400, 3), np.uint8)
        draw_debug_panel(frame, TextRenderer(), [1.0, 0.5, 0.0, 0.2, 0.9], [1, 1, 0, 0, 1], 10, 10)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
