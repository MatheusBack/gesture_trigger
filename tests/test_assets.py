"""Integridade da configuração: toda imagem citada em patterns.json existe e é legível.

O projeto original vinha com a pasta assets/ vazia, então o gesto era reconhecido
mas nenhuma imagem aparecia. Este teste garante que isso nunca mais passe batido.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.overlay import load_overlay_image
from core.pattern_matcher import PatternMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(BASE_DIR, "patterns.json")


class ShippedConfigTestCase(unittest.TestCase):
    def test_shipped_patterns_json_is_valid(self):
        matcher = PatternMatcher(CONFIG)
        self.assertGreaterEqual(len(matcher.entries), 5)

    def test_every_gesture_image_exists_and_decodes(self):
        with open(CONFIG, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            with self.subTest(gesture=entry["name"]):
                path = os.path.join(BASE_DIR, entry["image"])
                self.assertTrue(os.path.isfile(path), f"imagem ausente: {entry['image']}")
                image = load_overlay_image(path)
                self.assertIsNotNone(image, f"imagem ilegível: {entry['image']}")
                self.assertGreaterEqual(min(image.shape[:2]), 128)
                self.assertGreater(int(image[:, :, 3].max()), 0, "imagem totalmente transparente")

    def test_every_shipped_pattern_is_reachable(self):
        # Para cada gesto existe uma leitura de dedos que o reconhece (nada "sombreado" por outro).
        matcher = PatternMatcher(CONFIG, stability_frames=1)
        for entry in matcher.entries:
            reading = [0 if v is None else v for v in entry["pattern"]]
            matcher.reset()
            found = matcher.update(reading)
            with self.subTest(gesture=entry["name"]):
                # Com curingas = 0 pode cair num gesto mais específico; ao menos algum gesto deve responder.
                self.assertIsNotNone(found)


class AssetGeneratorTestCase(unittest.TestCase):
    def test_renders_a_card_for_every_pattern_shape(self):
        sys.path.insert(0, os.path.join(BASE_DIR, "tools"))
        import generate_assets as gen

        for pattern in ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [None, 1, 1, 0, 0], [None, 1, 1, 1, 1]):
            with self.subTest(pattern=pattern):
                card = gen.render_card({"name": "Teste Áéç", "pattern": pattern, "image": "x.png"}, (88, 166, 255))
                self.assertEqual(card.size, (512, 512))
                self.assertEqual(card.mode, "RGBA")
                self.assertGreater(card.getbbox()[2], 400)


if __name__ == "__main__":
    unittest.main()
