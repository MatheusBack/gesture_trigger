"""Elementos visuais: texto com acentos, esqueleto da mão e painel de debug."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.finger_state import FINGER_NAMES, FingerThresholds
from core.overlay import overlay_image

Color = Tuple[int, int, int]  # BGR

COLOR_EXTENDED: Color = (110, 220, 90)
COLOR_BENT: Color = (60, 120, 255)
COLOR_NEUTRAL: Color = (230, 230, 230)
COLOR_OK: Color = (120, 230, 120)
COLOR_WARN: Color = (90, 190, 255)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]  # fmt: skip


def _finger_of_point(index: int) -> Optional[int]:
    """0 = polegar ... 4 = mínimo; None para o punho."""
    return None if index == 0 else (index - 1) // 4


_REGULAR_FONTS = ("segoeui.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
_BOLD_FONTS = ("segoeuib.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")


def _load_font(size: int, bold: bool):
    for name in _BOLD_FONTS if bold else _REGULAR_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


class TextRenderer:
    """Desenha texto Unicode (acentos incluídos) sobre quadros do OpenCV.

    Cada texto é renderizado uma vez numa imagem BGRA pequena e guardado em cache;
    nos quadros seguintes só é composto, o que mantém o loop de vídeo rápido.
    """

    _CACHE_LIMIT = 256

    def __init__(self):
        self._fonts: Dict[Tuple[int, bool], ImageFont.ImageFont] = {}
        self._cache: Dict[Tuple[str, int, Color, bool], np.ndarray] = {}

    def render(self, text: str, size: int, color: Color = COLOR_NEUTRAL, bold: bool = False) -> np.ndarray:
        key = (text, size, color, bold)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        font = self._fonts.get((size, bold))
        if font is None:
            font = self._fonts[(size, bold)] = _load_font(size, bold)

        stroke = max(1, size // 12)
        left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
        pad = stroke + 1
        canvas = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (0, 0, 0, 0))
        rgb = (color[2], color[1], color[0])
        ImageDraw.Draw(canvas).text(
            (pad - left, pad - top), text, font=font, fill=rgb + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 230)
        )
        rgba = np.asarray(canvas)
        bgra = np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])

        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = bgra
        return bgra

    def draw(self, frame, text: str, x: int, y: int, size: int, color: Color = COLOR_NEUTRAL,
             bold: bool = False, align: str = "left"):
        """Desenha `text` com o canto superior esquerdo em (x, y); align="right" alinha à direita."""
        bgra = self.render(text, size, color, bold)
        w = bgra.shape[1]
        if align == "right":
            x -= w
        overlay_image(frame, bgra, position=(x, y), max_width=frame.shape[1], max_height=frame.shape[0])
        return bgra.shape[1], bgra.shape[0]


def draw_hand(frame, image_points: Sequence[Tuple[float, float]], states: Optional[Sequence[int]] = None):
    """Desenha o esqueleto da mão. Com `states`, cada dedo é verde (esticado) ou laranja (dobrado)."""
    pts = [(int(x), int(y)) for x, y in image_points]

    def color_for(point_index: int) -> Color:
        finger = _finger_of_point(point_index)
        if states is None or finger is None:
            return COLOR_NEUTRAL
        return COLOR_EXTENDED if states[finger] else COLOR_BENT

    palm = {0, 5, 9, 13, 17}  # linhas da palma não pertencem a nenhum dedo: ficam neutras
    for a, b in HAND_CONNECTIONS:
        color = COLOR_NEUTRAL if (a in palm and b in palm) else color_for(max(a, b))
        cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 5, color_for(i), 2, cv2.LINE_AA)
    return frame


def draw_debug_panel(frame, text: TextRenderer, scores: Sequence[float], states: Sequence[int],
                     x: int, y: int, thresholds: FingerThresholds = FingerThresholds(), size: int = 16):
    """Barras com a nota (0-1) de cada dedo e os limites de histerese.

    Se um gesto não é reconhecido, olhe aqui: a barra do dedo problemático mostra
    se a nota está perto do limite (ajuste as constantes em core/finger_state.py).
    """
    bar_w, bar_h, gap = 130, 10, size + 8
    for i, name in enumerate(FINGER_NAMES):
        row_y = y + i * gap
        text.draw(frame, name, x, row_y - 2, size)
        bx = x + 90
        cv2.rectangle(frame, (bx, row_y + 3), (bx + bar_w, row_y + 3 + bar_h), (40, 40, 40), -1)
        fill = int(bar_w * max(0.0, min(1.0, scores[i])))
        cv2.rectangle(frame, (bx, row_y + 3), (bx + fill, row_y + 3 + bar_h), COLOR_EXTENDED if states[i] else COLOR_BENT, -1)
        for limit in (thresholds.off, thresholds.on):
            lx = bx + int(bar_w * limit)
            cv2.line(frame, (lx, row_y + 1), (lx, row_y + 5 + bar_h), (255, 255, 255), 1)
        cv2.rectangle(frame, (bx, row_y + 3), (bx + bar_w, row_y + 3 + bar_h), (200, 200, 200), 1)
    return frame
