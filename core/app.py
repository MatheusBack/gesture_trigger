"""Orquestra um quadro de vídeo: detectar mão -> estimar dedos -> reconhecer -> desenhar.

Fica separado do loop da câmera (main.py) para que todo o fluxo possa ser testado
com quadros e um detector falsos, sem webcam.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.finger_state import FingerStateEstimator, FingerThresholds
from core.overlay import fit_within, load_overlay_image, overlay_image
from core.pattern_matcher import PatternMatcher
from core.ui import COLOR_NEUTRAL, COLOR_OK, COLOR_WARN, TextRenderer, draw_debug_panel, draw_hand


class ImageCache:
    """Carrega a imagem de cada gesto uma única vez; se faltar, avisa uma vez e segue."""

    def __init__(self, base_dir: str, warn: Callable[[str], None] = print):
        self._base_dir = base_dir
        self._warn = warn
        self._loaded: Dict[str, Optional[np.ndarray]] = {}
        self._fitted: Dict[Tuple[str, int, int], Optional[np.ndarray]] = {}

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self._base_dir, path)

    def get(self, image_path: str, max_width: int, max_height: int) -> Optional[np.ndarray]:
        key = (image_path, max_width, max_height)
        if key in self._fitted:
            return self._fitted[key]

        if image_path not in self._loaded:
            image = load_overlay_image(self._resolve(image_path))
            if image is None:
                self._warn(
                    f"[AVISO] Imagem não encontrada ou ilegível: {self._resolve(image_path)}\n"
                    "        Rode `python tools/generate_assets.py` para gerá-las, ou coloque seu PNG nesse caminho."
                )
            self._loaded[image_path] = image

        loaded = self._loaded[image_path]
        self._fitted[key] = None if loaded is None else fit_within(loaded, max_width, max_height)
        return self._fitted[key]


@dataclass
class FrameResult:
    frame: np.ndarray
    entry: Optional[dict]  # gesto ativo (estável) ou None
    changed: bool  # o gesto ativo mudou neste quadro?
    fingers: Optional[List[int]]  # leitura crua dos dedos neste quadro, ou None sem mão


class GestureApp:
    def __init__(
        self,
        detector,
        matcher: PatternMatcher,
        images: ImageCache,
        estimator: Optional[FingerStateEstimator] = None,
        text: Optional[TextRenderer] = None,
        show_landmarks: bool = True,
        debug: bool = False,
        fade_frames: int = 5,
    ):
        self._detector = detector
        self._matcher = matcher
        self._images = images
        self._estimator = estimator or FingerStateEstimator()
        self._text = text or TextRenderer()
        self.show_landmarks = show_landmarks
        self.debug = debug
        self._fade_frames = max(1, fade_frames)

        self._last_name: Optional[str] = None
        self._frames_shown = 0
        self._fps = 0.0
        self._last_tick: Optional[float] = None

    def toggle_debug(self) -> bool:
        self.debug = not self.debug
        return self.debug

    def toggle_landmarks(self) -> bool:
        self.show_landmarks = not self.show_landmarks
        return self.show_landmarks

    def process(self, frame: np.ndarray) -> FrameResult:
        """Processa um quadro BGR (já espelhado). Desenha nele e devolve o resultado."""
        self._tick()
        observation = self._detector.detect(frame)

        reading = None
        if observation is not None:
            reading = self._estimator.update(observation.points)
        else:
            self._estimator.reset()  # a mão sumiu: não carregar estado antigo para a próxima

        fingers = reading.states if reading else None
        entry = self._matcher.update(fingers)

        name = entry["name"] if entry else None
        changed = name != self._last_name
        self._last_name = name
        self._frames_shown = 0 if entry is None else (1 if changed else self._frames_shown + 1)

        if observation is not None and self.show_landmarks:
            draw_hand(frame, observation.image_points, fingers)

        self._draw_hud(frame, entry, observation is not None, reading)
        return FrameResult(frame=frame, entry=entry, changed=changed, fingers=fingers)

    # ------------------------------------------------------------------ desenho

    def _tick(self) -> None:
        now = time.perf_counter()
        if self._last_tick is not None and now > self._last_tick:
            instant = 1.0 / (now - self._last_tick)
            self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant
        self._last_tick = now

    def _draw_hud(self, frame, entry, hand_visible: bool, reading) -> None:
        h, w = frame.shape[:2]
        margin = max(10, h // 45)
        title_size = max(20, h // 18)
        small = max(13, h // 42)

        if entry is not None:
            self._text.draw(frame, entry["name"], margin, margin, title_size, COLOR_OK, bold=True)
            self._draw_card(frame, entry, margin, small)
        elif hand_visible:
            self._text.draw(frame, "Gesto não reconhecido", margin, margin, title_size - 4, COLOR_WARN)
        else:
            self._text.draw(frame, "Mostre a mão para a câmera", margin, margin, title_size - 4, COLOR_NEUTRAL)

        if self.debug and reading is not None:
            panel_h = 5 * (small + 8)
            draw_debug_panel(
                frame, self._text, reading.scores, reading.states,
                margin, h - margin - small - 12 - panel_h, FingerThresholds(), size=small,
            )

        self._text.draw(frame, "Q / ESC sair   D debug   L esqueleto", margin, h - margin - small - 4, small)
        self._text.draw(frame, f"{self._fps:4.0f} FPS", w - margin, h - margin - small - 4, small, align="right")

    def _draw_card(self, frame, entry, margin: int, small: int) -> None:
        h, w = frame.shape[:2]
        box = int(min(h * 0.46, w * 0.38))
        image = self._images.get(entry["image"], box, box)
        if image is None:
            self._text.draw(
                frame, f"(imagem não encontrada: {entry['image']})", margin, margin + max(20, h // 18) + 6, small, COLOR_WARN
            )
            return
        opacity = min(1.0, self._frames_shown / self._fade_frames)
        x = w - image.shape[1] - margin
        overlay_image(frame, image, position=(x, margin), max_width=box, max_height=box, opacity=opacity)
