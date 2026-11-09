"""Gera 21 pontos de uma mão sintética, com proporções anatômicas aproximadas.

Serve para testar o reconhecimento sem webcam: dá para esticar/dobrar cada dedo,
girar a mão em 3D, mudar a escala e adicionar ruído. Unidade: 1.0 = comprimento
da palma (punho -> base do dedo médio).
"""

import math
import random

# MCP de cada dedo (indicador, médio, anelar, mínimo): (x, y). z = 0 (plano da palma).
_MCP_XY = [(-0.33, 0.95), (-0.11, 1.00), (0.11, 0.95), (0.33, 0.85)]
# Comprimento das falanges (proximal, média, distal).
_SEGMENTS = [(0.40, 0.25, 0.20), (0.45, 0.28, 0.22), (0.41, 0.27, 0.22), (0.32, 0.18, 0.18)]
# Flexão máxima (graus) em MCP, PIP, DIP quando o dedo está totalmente dobrado.
_MAX_FLEX = (85.0, 100.0, 65.0)

# Polegar: cadeia CMC -> MCP -> IP -> ponta. Cada segmento tem uma abdução (graus a
# partir do eixo da palma, para fora) e uma flexão (graus, para dentro da palma).
_THUMB_SEGMENTS = (0.33, 0.31, 0.26)
_THUMB_CMC = (-0.22, 0.20, 0.0)
THUMB_POSES = {
    # mão aberta, polegar relaxado
    "open": ((55, 0), (50, 0), (40, 0)),
    # "joinha" / formato de L: polegar a ~80° do indicador
    "hitchhiker": ((75, 0), (80, 0), (85, 0)),
    # dobrado por cima da palma (punho com polegar cruzado)
    "across": ((35, 15), (0, 45), (-35, 80)),
    # dobrado, encostado na lateral do punho
    "alongside": ((20, 25), (10, 55), (0, 90)),
}


def _thumb_chain(pose):
    pts, pos = [_THUMB_CMC], _THUMB_CMC
    for length, (abd, flex) in zip(_THUMB_SEGMENTS, pose):
        a, f = math.radians(abd), math.radians(flex)
        d = (-math.sin(a) * math.cos(f), math.cos(a) * math.cos(f), -math.sin(f))
        pos = (pos[0] + length * d[0], pos[1] + length * d[1], pos[2] + length * d[2])
        pts.append(pos)
    return pts


def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def hand_points(curls, thumb_open="open", thumb_closed="across"):
    """curls: [polegar, indicador, médio, anelar, mínimo], 0.0 = esticado, 1.0 = dobrado.

    O polegar interpola entre as posturas `thumb_open` ("open", "hitchhiker") e
    `thumb_closed` ("across" = por cima da palma, "alongside" = colado ao punho).
    """
    pts = [(0.0, 0.0, 0.0)] * 21

    open_pts, closed_pts = _thumb_chain(THUMB_POSES[thumb_open]), _thumb_chain(THUMB_POSES[thumb_closed])
    for i, (a, b) in enumerate(zip(open_pts, closed_pts)):
        pts[1 + i] = _lerp(a, b, curls[0])

    for f in range(4):
        mcp = (_MCP_XY[f][0], _MCP_XY[f][1], 0.0)
        c = curls[1 + f]
        flex = [math.radians(m * c) for m in _MAX_FLEX]
        base = 5 + 4 * f
        pts[base] = mcp
        pos, angle = mcp, 0.0
        for seg in range(3):
            angle += flex[seg]
            # dobra no plano y-z, em direção à palma (z negativo)
            pos = (pos[0], pos[1] + _SEGMENTS[f][seg] * math.cos(angle), pos[2] - _SEGMENTS[f][seg] * math.sin(angle))
            pts[base + 1 + seg] = pos
    return pts


def transform(points, rx=0.0, ry=0.0, rz=0.0, scale=1.0, shift=(0.0, 0.0, 0.0)):
    """Gira (graus, em torno de x, y e z), escala e desloca todos os pontos."""
    ax, ay, az = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    out = []
    for x, y, z in points:
        y, z = y * cx - z * sx, y * sx + z * cx
        x, z = x * cy + z * sy, -x * sy + z * cy
        x, y = x * cz - y * sz, x * sz + y * cz
        out.append((x * scale + shift[0], y * scale + shift[1], z * scale + shift[2]))
    return out


def add_noise(points, sigma, rng=None):
    """Ruído gaussiano de desvio `sigma` (em tamanhos de palma) em cada coordenada."""
    rng = rng or random.Random(0)
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), z + rng.gauss(0, sigma)) for x, y, z in points]


# Posturas prontas: (curls polegar..mínimo, polegar esticado, polegar dobrado).
POSES = {
    "punho": ([1, 1, 1, 1, 1], "open", "across"),
    "punho_polegar_lateral": ([1, 1, 1, 1, 1], "open", "alongside"),
    "indicador": ([1, 0, 1, 1, 1], "open", "across"),
    "positivo": ([0, 1, 1, 1, 1], "hitchhiker", "across"),
    "paz": ([1, 0, 0, 1, 1], "open", "across"),
    "mao_aberta": ([0, 0, 0, 0, 0], "open", "across"),
    "mao_aberta_L": ([0, 0, 0, 0, 0], "hitchhiker", "across"),
}

EXPECTED = {
    "punho": [0, 0, 0, 0, 0],
    "punho_polegar_lateral": [0, 0, 0, 0, 0],
    "mao_aberta_L": [1, 1, 1, 1, 1],
    "indicador": [0, 1, 0, 0, 0],
    "positivo": [1, 0, 0, 0, 0],
    "paz": [0, 1, 1, 0, 0],
    "mao_aberta": [1, 1, 1, 1, 1],
}


def pose_points(name, **transform_kwargs):
    curls, thumb_open, thumb_closed = POSES[name]
    return transform(hand_points(curls, thumb_open, thumb_closed), **transform_kwargs)
