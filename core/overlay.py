"""Desenha uma imagem (com ou sem transparência) por cima de um quadro de vídeo."""

from __future__ import annotations

import cv2
import numpy as np


def _to_bgra(image: np.ndarray) -> np.ndarray:
    """Converte qualquer imagem do OpenCV (cinza, BGR, BGRA; 8 ou 16 bits) para BGRA uint8."""
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    if channels == 4:
        return image
    raise ValueError(f"Número de canais não suportado: {channels}")


def load_overlay_image(image_path: str):
    """Lê uma imagem do disco como BGRA uint8. Devolve None se ilegível ou inexistente.

    Usa `np.fromfile` + `cv2.imdecode` em vez de `cv2.imread` porque `imread` não
    consegue abrir caminhos com acento no Windows (ex.: C:\\Users\\João\\Projeto).
    """
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return None if decoded is None else _to_bgra(decoded)
    except (OSError, ValueError, cv2.error):
        return None


def fit_within(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Reduz a imagem para caber na caixa, mantendo a proporção. Nunca amplia."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_image(frame, overlay_img, position=(10, 10), max_width=220, max_height=220, opacity=1.0):
    """Compõe `overlay_img` sobre `frame` (modificado no lugar) e devolve o quadro.

    * Respeita transparência (canal alfa) e aceita imagens cinza, BGR ou BGRA.
    * Reduz para caber em max_width x max_height sem distorcer.
    * Recorta com segurança se passar da borda (inclusive para posições negativas).
    * `opacity` (0 a 1) multiplica o alfa: serve para fade-in/fade-out.
    """
    if overlay_img is None or opacity <= 0.0:
        return frame

    overlay = fit_within(_to_bgra(overlay_img), max_width, max_height)
    x, y = int(position[0]), int(position[1])
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]

    # Interseção entre o retângulo da imagem e o do quadro.
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + ow, fw), min(y + oh, fh)
    if x1 <= x0 or y1 <= y0:
        return frame
    overlay = overlay[y0 - y : y1 - y, x0 - x : x1 - x]

    alpha = overlay[:, :, 3:4].astype(np.float32) * (min(opacity, 1.0) / 255.0)
    roi = frame[y0:y1, x0:x1]
    blended = overlay[:, :, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)
    return frame
