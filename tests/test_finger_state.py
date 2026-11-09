"""Reconhecimento de dedos: geometria 3D, rotação, ruído e histerese."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synthetic_hand as sh
from core.finger_state import FingerStateEstimator, FingerThresholds, finger_scores, fingers_up_from_points


class FingerScoresTestCase(unittest.TestCase):
    def test_each_pose_is_read_correctly(self):
        for name, expected in sh.EXPECTED.items():
            with self.subTest(pose=name):
                self.assertEqual(fingers_up_from_points(sh.pose_points(name)), expected)

    def test_result_does_not_depend_on_hand_orientation(self):
        # O bug central do código antigo: comparar y só funciona com a mão em pé.
        rng = random.Random(1)
        for _ in range(150):
            name = rng.choice(list(sh.EXPECTED))
            pts = sh.pose_points(
                name, rx=rng.uniform(-60, 60), ry=rng.uniform(-60, 60), rz=rng.uniform(-180, 180)
            )
            self.assertEqual(fingers_up_from_points(pts), sh.EXPECTED[name], f"pose={name}")

    def test_result_does_not_depend_on_distance_from_camera(self):
        for scale in (0.3, 1.0, 4.0):
            for name, expected in sh.EXPECTED.items():
                pts = sh.pose_points(name, scale=scale, shift=(5, -3, 2))
                self.assertEqual(fingers_up_from_points(pts), expected, f"pose={name} scale={scale}")

    def test_mirrored_hand_gives_same_result(self):
        # Mão esquerda = espelho da direita. Sem lógica de lateralidade, deve ser igual.
        for name, expected in sh.EXPECTED.items():
            mirrored = [(-x, y, z) for x, y, z in sh.pose_points(name)]
            self.assertEqual(fingers_up_from_points(mirrored), expected, f"pose={name}")

    def test_thumb_tucked_alongside_fist_is_not_extended(self):
        self.assertEqual(fingers_up_from_points(sh.pose_points("punho_polegar_lateral"))[0], 0)

    def test_scores_stay_in_unit_range_and_are_monotonic_with_curl(self):
        previous = 2.0
        for curl in [i / 10 for i in range(11)]:
            score = finger_scores(sh.hand_points([1, curl, 1, 1, 1]))[1]
            self.assertTrue(0.0 <= score <= 1.0)
            self.assertLessEqual(score, previous + 1e-9, "dobrar mais não pode aumentar a nota")
            previous = score

    def test_degenerate_hand_does_not_crash(self):
        self.assertEqual(finger_scores([(0.0, 0.0, 0.0)] * 21), [0.0] * 5)

    def test_wrong_number_of_points_is_rejected(self):
        with self.assertRaises(ValueError):
            finger_scores([(0, 0, 0)] * 20)


class FingerStateEstimatorTestCase(unittest.TestCase):
    def test_stays_correct_under_detector_noise(self):
        for name, expected in sh.EXPECTED.items():
            est, rng = FingerStateEstimator(), random.Random(7)
            base = sh.pose_points(name, rx=20, ry=-15, rz=30)
            correct = 0
            for frame in range(60):
                reading = est.update(sh.add_noise(base, 0.03, rng))
                if frame >= 5:
                    correct += reading.states == expected
            self.assertGreaterEqual(correct / 55, 0.95, f"pose={name}")

    def test_hysteresis_prevents_flicker_near_the_threshold(self):
        # Alterna entre duas leituras logo abaixo/acima de 0.5. Sem histerese o estado
        # trocaria a cada quadro e o gesto nunca ficaria estável.
        est = FingerStateEstimator(FingerThresholds(on=0.62, off=0.38, smoothing=1.0))
        wobble = [sh.hand_points([1, c, 1, 1, 1]) for c in (0.38, 0.42, 0.38, 0.42)]
        states = [est.update(p).states[1] for p in wobble * 5]
        self.assertEqual(len(set(states)), 1, "estado oscilou entre quadros")

    def test_state_changes_when_finger_clearly_moves(self):
        est = FingerStateEstimator(FingerThresholds(smoothing=1.0))
        self.assertEqual(est.update(sh.hand_points([1, 1, 1, 1, 1])).states[1], 0)
        self.assertEqual(est.update(sh.hand_points([1, 0, 1, 1, 1])).states[1], 1)
        self.assertEqual(est.update(sh.hand_points([1, 1, 1, 1, 1])).states[1], 0)

    def test_reset_forgets_previous_state(self):
        est = FingerStateEstimator()
        est.update(sh.pose_points("mao_aberta"))
        est.reset()
        self.assertEqual(est.update(sh.pose_points("punho")).states, [0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
