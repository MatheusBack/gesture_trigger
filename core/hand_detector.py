"""Encontra a mão no vídeo usando o MediaPipe HandLandmarker (API Tasks).

A API antiga `mp.solutions.hands` foi removida das versões recentes do MediaPipe,
por isso o projeto usa a API Tasks, que precisa de um arquivo de modelo
(`hand_landmarker.task`, ~8 MB). Ele é baixado sozinho na primeira execução.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2

from core.finger_state import Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
_MIN_MODEL_BYTES = 1_000_000


class ModelUnavailableError(RuntimeError):
    """O arquivo de modelo não existe e não foi possível baixá-lo."""


@dataclass
class HandObservation:
    """O que o detector enxergou em um quadro."""

    points: List[Point]  # 21 pontos 3D, usados na geometria dos dedos
    image_points: List[Tuple[float, float]]  # 21 pontos em pixels, usados para desenhar


def ensure_model(
    path: str = DEFAULT_MODEL_PATH,
    url: str = MODEL_URL,
    downloader: Optional[Callable[[str, str], object]] = None,
) -> str:
    """Garante que o modelo existe em `path`, baixando-o se necessário."""
    if os.path.isfile(path) and os.path.getsize(path) >= _MIN_MODEL_BYTES:
        return path

    manual = (
        f"Baixe manualmente:\n  {url}\ne salve como:\n  {path}\n"
        "(ou use --model CAMINHO para apontar para outro arquivo)."
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".part"
    print(f"Baixando o modelo de detecção de mão (uma vez só, ~8 MB)...\n  {url}")
    try:
        (downloader or urllib.request.urlretrieve)(url, partial)
        if os.path.getsize(partial) < _MIN_MODEL_BYTES:
            raise OSError("arquivo baixado é pequeno demais")
        with open(partial, "rb") as f:
            head = f.read(64).lstrip().lower()
        if head.startswith((b"<!doctype", b"<html", b"<?xml", b"{")):
            raise OSError("o servidor devolveu uma página de erro em vez do modelo (proxy, antivírus ou rede?)")
        os.replace(partial, path)
    except Exception as exc:  # noqa: BLE001 - qualquer falha vira uma mensagem útil
        if os.path.exists(partial):
            os.remove(partial)
        raise ModelUnavailableError(f"Não foi possível baixar o modelo ({exc}).\n{manual}") from exc
    return path


def observation_from_result(result, width: int, height: int) -> Optional[HandObservation]:
    """Converte o resultado do MediaPipe em HandObservation (ou None se não há mão).

    Prefere os `hand_world_landmarks` (coordenadas 3D em metros, sem distorção de
    proporção da imagem). Se não vierem, usa os pontos normalizados da imagem
    convertidos para pixels, para que x, y e z fiquem na mesma escala.
    """
    if not getattr(result, "hand_landmarks", None):
        return None

    landmarks = result.hand_landmarks[0]
    image_points = [(p.x * width, p.y * height) for p in landmarks]

    world = getattr(result, "hand_world_landmarks", None)
    if world:
        points = [(p.x, p.y, p.z) for p in world[0]]
    else:
        points = [(p.x * width, p.y * height, p.z * width) for p in landmarks]
    if len(points) < 21:
        return None
    return HandObservation(points=points, image_points=image_points)


class HandDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # Importa aqui (e não no topo) para que o resto do projeto e seus testes
        # funcionem sem MediaPipe, e para dar uma mensagem clara se faltar.
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe não está instalado. Rode: pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        resolved = ensure_model(model_path or DEFAULT_MODEL_PATH)
        with open(resolved, "rb") as f:  # bytes em vez de caminho: evita falhas com acentos no Windows
            model_bytes = f.read()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"O MediaPipe não conseguiu carregar o modelo ({exc}).\n"
                f"Apague o arquivo {resolved} e rode de novo para baixá-lo outra vez."
            ) from exc
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr) -> Optional[HandObservation]:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # O modo VIDEO exige timestamps estritamente crescentes.
        timestamp = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp

        result = self._landmarker.detect_for_video(image, timestamp)
        return observation_from_result(result, width, height)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
