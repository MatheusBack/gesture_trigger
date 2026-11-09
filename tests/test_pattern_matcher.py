"""PatternMatcher: reconhecimento, debounce, curingas e erros de configuração."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pattern_matcher import PatternMatcher

FIST = [0, 0, 0, 0, 0]
OPEN_HAND = [1, 1, 1, 1, 1]
PEACE = [0, 1, 1, 0, 0]
THUMBS_UP = [1, 0, 0, 0, 0]

SAMPLE_CONFIG = [
    {"name": "Punho Fechado", "pattern": FIST, "image": "assets/fist.png"},
    {"name": "Mao Aberta", "pattern": OPEN_HAND, "image": "assets/open.png"},
    {"name": "Sinal de Paz", "pattern": PEACE, "image": "assets/peace.png"},
]


def write_config(data):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


class PatternMatcherTestCase(unittest.TestCase):
    def setUp(self):
        self.config_path = write_config(SAMPLE_CONFIG)
        self.addCleanup(os.remove, self.config_path)

    def _matcher(self, stability_frames=3, release_frames=1):
        return PatternMatcher(self.config_path, stability_frames, release_frames)

    def test_pattern_not_active_before_reaching_stability_threshold(self):
        matcher = self._matcher(3)
        self.assertIsNone(matcher.update(FIST))
        self.assertIsNone(matcher.update(FIST))

    def test_pattern_becomes_active_once_stable(self):
        matcher = self._matcher(3)
        matcher.update(FIST)
        matcher.update(FIST)
        self.assertEqual(matcher.update(FIST)["name"], "Punho Fechado")

    def test_flicker_resets_the_stability_counter(self):
        matcher = self._matcher(3)
        matcher.update(FIST)
        matcher.update(FIST)
        matcher.update(OPEN_HAND)
        self.assertIsNone(matcher.update(FIST))

    def test_switching_between_two_known_patterns(self):
        matcher = self._matcher(2)
        matcher.update(FIST)
        self.assertEqual(matcher.update(FIST)["name"], "Punho Fechado")
        matcher.update(PEACE)
        self.assertEqual(matcher.update(PEACE)["name"], "Sinal de Paz")

    def test_unrecognized_pattern_returns_none(self):
        self.assertIsNone(self._matcher(1).update([1, 0, 1, 0, 1]))

    # --- o bug do gesto "preso" na tela -------------------------------------

    def test_gesture_is_released_after_release_frames_without_hand(self):
        matcher = self._matcher(2, release_frames=3)
        matcher.update(FIST)
        matcher.update(FIST)
        self.assertIsNotNone(matcher.update(None))  # 1 quadro sem mão: ainda segura
        self.assertIsNotNone(matcher.update(None))
        self.assertIsNone(matcher.update(None))  # 3º quadro seguido: solta

    def test_single_dropped_frame_does_not_make_image_blink(self):
        matcher = self._matcher(2, release_frames=3)
        matcher.update(FIST)
        matcher.update(FIST)
        matcher.update(None)  # a detecção falhou por 1 quadro
        self.assertEqual(matcher.update(FIST)["name"], "Punho Fechado")
        self.assertEqual(matcher.update(None)["name"], "Punho Fechado")  # contador recomeçou

    def test_gesture_is_released_when_hand_changes_to_unknown_pose(self):
        matcher = self._matcher(1, release_frames=2)
        matcher.update(FIST)
        matcher.update([1, 0, 1, 0, 1])
        self.assertIsNone(matcher.update([1, 0, 1, 0, 1]))

    def test_reset_clears_active_gesture(self):
        matcher = self._matcher(1)
        matcher.update(FIST)
        matcher.reset()
        self.assertIsNone(matcher.update(None))

    # --- curingas ------------------------------------------------------------

    def test_wildcard_ignores_that_finger(self):
        path = write_config([{"name": "Paz", "pattern": [None, 1, 1, 0, 0], "image": "a.png"}])
        self.addCleanup(os.remove, path)
        matcher = PatternMatcher(path, stability_frames=1)
        self.assertEqual(matcher.update([0, 1, 1, 0, 0])["name"], "Paz")
        matcher.reset()
        self.assertEqual(matcher.update([1, 1, 1, 0, 0])["name"], "Paz")

    def test_most_specific_pattern_wins_over_wildcard(self):
        path = write_config(
            [
                {"name": "Punho", "pattern": [None, 0, 0, 0, 0], "image": "a.png"},
                {"name": "Positivo", "pattern": THUMBS_UP, "image": "b.png"},
            ]
        )
        self.addCleanup(os.remove, path)
        matcher = PatternMatcher(path, stability_frames=1)
        self.assertEqual(matcher.update(THUMBS_UP)["name"], "Positivo")
        matcher.reset()
        self.assertEqual(matcher.update(FIST)["name"], "Punho")

    # --- configuração inválida ----------------------------------------------

    def test_missing_config_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            PatternMatcher("this/path/does/not/exist.json")

    def test_malformed_config_entry_raises(self):
        path = write_config([{"name": "Incompleto"}])
        self.addCleanup(os.remove, path)
        with self.assertRaises(KeyError):
            PatternMatcher(path)

    def test_pattern_with_wrong_length_raises_clear_error(self):
        path = write_config([{"name": "X", "pattern": [1, 0, 1], "image": "a.png"}])
        self.addCleanup(os.remove, path)
        with self.assertRaisesRegex(ValueError, "5 valores"):
            PatternMatcher(path)

    def test_pattern_with_invalid_value_raises(self):
        path = write_config([{"name": "X", "pattern": [0, 2, 0, 0, 0], "image": "a.png"}])
        self.addCleanup(os.remove, path)
        with self.assertRaisesRegex(ValueError, "0, 1 ou null"):
            PatternMatcher(path)

    def test_duplicate_patterns_raise_instead_of_silently_shadowing(self):
        path = write_config(
            [
                {"name": "A", "pattern": FIST, "image": "a.png"},
                {"name": "B", "pattern": FIST, "image": "b.png"},
            ]
        )
        self.addCleanup(os.remove, path)
        with self.assertRaisesRegex(ValueError, "mesmo padrão"):
            PatternMatcher(path)


if __name__ == "__main__":
    unittest.main()
