"""Overlay: composição, transparência, bordas e leitura de arquivos."""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.overlay import fit_within, load_overlay_image, overlay_image


def solid_bgr(width, height, color):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def solid_bgra(width, height, color, alpha):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = alpha
    return img


class OverlayImageTestCase(unittest.TestCase):
    def setUp(self):
        self.frame = solid_bgr(400, 300, (50, 50, 50))

    def draw(self, overlay, **kwargs):
        kwargs.setdefault("max_width", 100)
        kwargs.setdefault("max_height", 100)
        return overlay_image(self.frame.copy(), overlay, **kwargs)

    def test_opaque_bgr_overlay_is_drawn_at_position(self):
        result = self.draw(solid_bgr(50, 50, (0, 255, 0)), position=(10, 10))
        np.testing.assert_array_equal(result[30, 30], [0, 255, 0])

    def test_fully_transparent_overlay_keeps_background(self):
        result = self.draw(solid_bgra(50, 50, (0, 0, 255), 0), position=(10, 10))
        np.testing.assert_array_equal(result[30, 30], [50, 50, 50])

    def test_fully_opaque_alpha_overlay_replaces_background(self):
        result = self.draw(solid_bgra(50, 50, (0, 0, 255), 255), position=(10, 10))
        np.testing.assert_array_equal(result[30, 30], [0, 0, 255])

    def test_half_alpha_blends(self):
        result = self.draw(solid_bgra(50, 50, (250, 250, 250), 128), position=(0, 0))
        self.assertTrue(140 <= int(result[10, 10, 0]) <= 160)

    def test_opacity_parameter_fades_the_overlay(self):
        full = self.draw(solid_bgr(50, 50, (250, 250, 250)), position=(0, 0), opacity=1.0)
        half = self.draw(solid_bgr(50, 50, (250, 250, 250)), position=(0, 0), opacity=0.5)
        none = self.draw(solid_bgr(50, 50, (250, 250, 250)), position=(0, 0), opacity=0.0)
        self.assertEqual(int(full[5, 5, 0]), 250)
        self.assertTrue(140 <= int(half[5, 5, 0]) <= 160)
        self.assertEqual(int(none[5, 5, 0]), 50)

    def test_aspect_ratio_is_preserved_when_downscaling(self):
        result = self.draw(solid_bgr(400, 100, (0, 255, 0)), position=(0, 0))
        np.testing.assert_array_equal(result[10, 10], [0, 255, 0])
        np.testing.assert_array_equal(result[90, 10], [50, 50, 50])  # só ~25 px de altura foram usados

    def test_small_image_is_never_upscaled(self):
        self.assertEqual(fit_within(solid_bgr(20, 20, (1, 2, 3)), 200, 200).shape[:2], (20, 20))

    def test_overlay_past_the_right_and_bottom_edge_is_clipped(self):
        result = self.draw(solid_bgr(100, 100, (0, 255, 0)), position=(380, 290))
        self.assertEqual(result.shape, self.frame.shape)
        np.testing.assert_array_equal(result[295, 390], [0, 255, 0])

    def test_negative_position_is_clipped_not_crashed(self):
        # No código original isto lançava ValueError (broadcast).
        result = self.draw(solid_bgr(50, 50, (255, 255, 255)), position=(-20, 10), max_width=50, max_height=50)
        self.assertEqual(int((result[:, :, 0] == 255).sum()), 30 * 50)

    def test_completely_outside_frame_is_ignored(self):
        np.testing.assert_array_equal(self.draw(solid_bgr(50, 50, (0, 255, 0)), position=(1000, 1000)), self.frame)

    def test_grayscale_overlay_does_not_crash(self):
        # No código original: IndexError (imagem 2D não tem eixo de canais).
        gray = np.full((40, 40), 200, dtype=np.uint8)
        result = self.draw(gray, position=(0, 0))
        np.testing.assert_array_equal(result[10, 10], [200, 200, 200])

    def test_none_overlay_returns_frame_unchanged(self):
        np.testing.assert_array_equal(overlay_image(self.frame.copy(), None), self.frame)


class LoadOverlayImageTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write_png(self, name, array):
        path = os.path.join(self.dir, name)
        ok, buf = cv2.imencode(".png", array)
        self.assertTrue(ok)
        buf.tofile(path)  # funciona também com nomes acentuados
        return path

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_overlay_image(os.path.join(self.dir, "nao_existe.png")))

    def test_corrupt_file_returns_none(self):
        path = os.path.join(self.dir, "lixo.png")
        with open(path, "wb") as f:
            f.write(b"isto nao e uma imagem")
        self.assertIsNone(load_overlay_image(path))

    def test_empty_file_returns_none(self):
        path = os.path.join(self.dir, "vazio.png")
        open(path, "wb").close()
        self.assertIsNone(load_overlay_image(path))

    def test_always_returns_bgra_uint8(self):
        cases = {
            "cinza.png": np.full((20, 30), 90, np.uint8),
            "bgr.png": solid_bgr(30, 20, (1, 2, 3)),
            "bgra.png": solid_bgra(30, 20, (1, 2, 3), 77),
            "16bits.png": np.full((20, 30), 40000, np.uint16),
        }
        for name, array in cases.items():
            with self.subTest(file=name):
                image = load_overlay_image(self.write_png(name, array))
                self.assertEqual(image.shape, (20, 30, 4))
                self.assertEqual(image.dtype, np.uint8)

    def test_accented_path_is_readable(self):
        folder = os.path.join(self.dir, "João Pereira")
        os.makedirs(folder)
        path = os.path.join(folder, "imagem ção.png")
        cv2.imencode(".png", solid_bgr(10, 10, (9, 9, 9)))[1].tofile(path)
        self.assertIsNotNone(load_overlay_image(path))


if __name__ == "__main__":
    unittest.main()
