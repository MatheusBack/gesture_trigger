"""Visual Pattern Gesture Trigger.

Abre a webcam, encontra a mão, reconhece o gesto (quais dedos estão esticados) e,
quando ele se mantém estável por alguns quadros, mostra a imagem correspondente
por cima do vídeo e avisa no terminal.

Uso:  python main.py [--camera N] [--debug] [--help]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import cv2

from core.app import GestureApp, ImageCache
from core.finger_state import FingerStateEstimator
from core.hand_detector import DEFAULT_MODEL_PATH, HandDetector, ModelUnavailableError
from core.pattern_matcher import PatternMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW = "Visual Pattern Gesture Trigger"
MAX_CONSECUTIVE_READ_FAILURES = 60


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reconhece gestos de mão pela webcam e mostra uma imagem para cada um.")
    p.add_argument("--camera", type=int, default=0, help="índice da webcam (0, 1, 2...). Padrão: 0")
    p.add_argument("--config", default=os.path.join(BASE_DIR, "patterns.json"), help="arquivo de gestos")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="caminho do modelo hand_landmarker.task")
    p.add_argument("--width", type=int, default=1280, help="largura pedida à câmera. Padrão: 1280")
    p.add_argument("--height", type=int, default=720, help="altura pedida à câmera. Padrão: 720")
    p.add_argument("--stability", type=int, default=5, help="quadros para confirmar um gesto. Padrão: 5")
    p.add_argument("--release", type=int, default=4, help="quadros para dispensar um gesto. Padrão: 4")
    p.add_argument("--debug", action="store_true", help="começa com o painel de notas dos dedos ligado")
    p.add_argument("--no-landmarks", action="store_true", help="não desenha o esqueleto da mão")
    return p.parse_args(argv)


def open_camera(index: int, width: int, height: int):
    """Abre a webcam. No Windows tenta DirectShow primeiro (o backend padrão costuma
    abrir devagar ou devolver quadros pretos). Devolve None se nenhum backend funcionar."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos atraso entre a mão e a tela
        for _ in range(40):  # muitas câmeras entregam quadros vazios no começo
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.05)
        cap.release()
    return None


def window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        matcher = PatternMatcher(args.config, stability_frames=args.stability, release_frames=args.release)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[ERRO] Configuração de gestos inválida: {exc}")
        return 1

    try:
        detector = HandDetector(model_path=args.model)
    except (ModelUnavailableError, RuntimeError) as exc:
        print(f"[ERRO] {exc}")
        return 1

    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        detector.close()
        print(
            f"[ERRO] Não consegui abrir a webcam (índice {args.camera}).\n"
            "  - Feche outros programas que usam a câmera (Teams, Zoom, navegador).\n"
            "  - Tente outro índice: python main.py --camera 1\n"
            "  - No Windows, confira Configurações > Privacidade > Câmera."
        )
        return 1

    app = GestureApp(
        detector,
        matcher,
        ImageCache(BASE_DIR),
        estimator=FingerStateEstimator(),
        show_landmarks=not args.no_landmarks,
        debug=args.debug,
    )

    print("Pronto. Faça um gesto para a câmera. Q ou ESC fecham; D liga o debug; L liga/desliga o esqueleto.")
    failures = 0
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > MAX_CONSECUTIVE_READ_FAILURES:
                    print("[ERRO] A webcam parou de enviar imagens.")
                    return 1
                time.sleep(0.01)
                continue
            failures = 0

            result = app.process(cv2.flip(frame, 1))  # espelhado, como um espelho de verdade
            if result.changed and result.entry is not None:
                print(f"[{datetime.now():%H:%M:%S}] Gesto reconhecido: {result.entry['name']}")

            cv2.imshow(WINDOW, result.frame)
            frames += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                app.toggle_debug()
            if key in (ord("l"), ord("L")):
                app.toggle_landmarks()
            if frames > 5 and window_closed():  # clicou no X da janela
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
