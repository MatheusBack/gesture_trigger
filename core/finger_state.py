"""Decide quais dedos estão esticados a partir dos 21 pontos 3D da mão.

Por que não basta comparar a altura (y) da ponta com a da junta?
Essa regra só funciona com a mão em pé, de frente para a câmera. Basta inclinar
ou girar a mão e ela erra. Aqui usamos **ângulos entre as articulações** e
**distâncias normalizadas pelo tamanho da palma**: valores que não mudam quando
a mão gira, se afasta da câmera ou é esquerda/direita.

Este módulo é geometria pura (só `math`), sem MediaPipe nem OpenCV, então é
totalmente testável sem câmera.

Cada ponto é uma tupla (x, y, z). A numeração segue o MediaPipe Hands:

    0 punho
    1-4   polegar   (CMC, MCP, IP, ponta)
    5-8   indicador (MCP, PIP, DIP, ponta)
    9-12  médio
    13-16 anelar
    17-20 mínimo
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

Point = Tuple[float, float, float]

FINGER_NAMES = ("Polegar", "Indicador", "Médio", "Anelar", "Mínimo")

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
# (MCP, PIP, ponta) de indicador, médio, anelar e mínimo
_LONG_FINGERS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))

# Faixas de calibração. Ângulo da articulação PIP: dedo reto ≈ 170-180°,
# dedo dobrado no punho ≈ 50-90°.
_PIP_ANGLE_BENT, _PIP_ANGLE_STRAIGHT = 100.0, 160.0
# Alcance = distância(punho, ponta) / distância(punho, MCP). Dobrado ≈ 1.0-1.3,
# esticado ≈ 1.8-2.0.
_REACH_BENT, _REACH_STRAIGHT = 1.30, 1.70
# Polegar: (1) ângulo da articulação IP e (2) "saída lateral": quanto a ponta
# passa da linha dos nós dos dedos, para o lado do polegar, em tamanhos de palma.
# Dobrado/encostado na palma ≈ -0.3 a 0.1; esticado ≈ 0.4 a 0.8.
_THUMB_ANGLE_BENT, _THUMB_ANGLE_STRAIGHT = 115.0, 150.0
_THUMB_OUT_BENT, _THUMB_OUT_OPEN = 0.18, 0.38


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Point) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dist(a: Point, b: Point) -> float:
    return _norm(_sub(a, b))


def _angle_deg(a: Point, b: Point, c: Point) -> float:
    """Ângulo em graus no vértice b, formado por a-b-c."""
    ba, bc = _sub(a, b), _sub(c, b)
    denom = _norm(ba) * _norm(bc)
    if denom < 1e-9:
        return 180.0
    cos = (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _ramp(value: float, low: float, high: float) -> float:
    """0 quando value <= low, 1 quando value >= high, linear no meio."""
    if high == low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def palm_size(points: Sequence[Point]) -> float:
    """Distância punho -> base do dedo médio. Serve de régua da mão."""
    return _dist(points[WRIST], points[MIDDLE_MCP])


def finger_scores(points: Sequence[Point]) -> List[float]:
    """Devolve 5 notas de 0.0 (dobrado) a 1.0 (esticado): polegar ... mínimo.

    Cada nota combina duas medidas independentes (ângulo e alcance) e usa a
    mais pessimista, então um dedo só conta como esticado se *ambas* concordam.
    """
    if len(points) < 21:
        raise ValueError(f"Esperava 21 pontos, recebi {len(points)}")

    size = palm_size(points)
    if size < 1e-9:
        return [0.0] * 5

    scores = [_thumb_score(points, size)]
    for mcp, pip, tip in _LONG_FINGERS:
        angle = _ramp(
            _angle_deg(points[mcp], points[pip], points[tip]),
            _PIP_ANGLE_BENT,
            _PIP_ANGLE_STRAIGHT,
        )
        base = _dist(points[WRIST], points[mcp])
        reach = _dist(points[WRIST], points[tip]) / base if base > 1e-9 else 0.0
        scores.append(min(angle, _ramp(reach, _REACH_BENT, _REACH_STRAIGHT)))
    return scores


def _thumb_score(points: Sequence[Point], size: float) -> float:
    # O polegar se move em outro plano que os demais dedos, então tem regra
    # própria: precisa estar reto E sair para o lado da palma. A "saída lateral"
    # é medida na direção indicador -> mínimo (sentido invertido), que aponta
    # sempre para o lado do polegar, seja a mão esquerda ou direita: nenhuma
    # informação de lateralidade é necessária.
    straight = _ramp(
        _angle_deg(points[THUMB_MCP], points[THUMB_IP], points[THUMB_TIP]),
        _THUMB_ANGLE_BENT,
        _THUMB_ANGLE_STRAIGHT,
    )
    knuckle_line = _sub(points[PINKY_MCP], points[INDEX_MCP])
    length = _norm(knuckle_line)
    if length < 1e-9:
        return 0.0
    to_tip = _sub(points[THUMB_TIP], points[INDEX_MCP])
    out = -(
        to_tip[0] * knuckle_line[0] + to_tip[1] * knuckle_line[1] + to_tip[2] * knuckle_line[2]
    ) / (length * size)
    return min(straight, _ramp(out, _THUMB_OUT_BENT, _THUMB_OUT_OPEN))


def fingers_up_from_points(points: Sequence[Point]) -> List[int]:
    """Versão sem memória: corta cada nota em 0.5. Útil em testes e scripts."""
    return [1 if s >= 0.5 else 0 for s in finger_scores(points)]


@dataclass(frozen=True)
class FingerThresholds:
    on: float = 0.62  # nota acima disso -> dedo passa a "esticado"
    off: float = 0.38  # nota abaixo disso -> dedo passa a "dobrado"
    smoothing: float = 0.55  # peso da leitura nova na média (1.0 = sem suavizar)


@dataclass
class FingerReading:
    states: List[int]
    scores: List[float]


@dataclass
class FingerStateEstimator:
    """Transforma notas ruidosas em estados 0/1 estáveis.

    O sintoma típico de "o reconhecimento falha" é um dedo com nota perto do
    limite que fica trocando 0,1,0,1 a cada quadro: isso zera o contador de
    estabilidade e o gesto nunca é confirmado. Duas defesas:

    * média móvel exponencial das notas (amortece o tremor do detector);
    * histerese: para mudar de estado a nota precisa cruzar `on`/`off`, e entre
      os dois limites o dedo mantém o estado anterior.
    """

    thresholds: FingerThresholds = field(default_factory=FingerThresholds)
    _smoothed: List[float] = field(default_factory=list, init=False, repr=False)
    _states: List[int] = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        self._smoothed = []
        self._states = []

    def update(self, points: Sequence[Point]) -> FingerReading:
        raw = finger_scores(points)
        t = self.thresholds

        if not self._smoothed:
            self._smoothed = list(raw)
            self._states = [1 if s >= 0.5 else 0 for s in raw]
        else:
            self._smoothed = [
                t.smoothing * new + (1.0 - t.smoothing) * old
                for new, old in zip(raw, self._smoothed)
            ]
            for i, score in enumerate(self._smoothed):
                if score >= t.on:
                    self._states[i] = 1
                elif score <= t.off:
                    self._states[i] = 0

        return FingerReading(states=list(self._states), scores=list(self._smoothed))
