"""Gera um cartão PNG com a ilustração de cada gesto de patterns.json.

    python tools/generate_assets.py            # cria só as imagens que faltam
    python tools/generate_assets.py --force    # recria todas (sobrescreve!)

A ilustração é desenhada a partir do próprio `pattern` do gesto, então a imagem
sempre bate com o que o reconhecedor espera. Para usar uma imagem sua (foto do
sinal em Libras, ícone, etc.), basta salvar um PNG no caminho indicado em
`image` e rodar sem --force: o seu arquivo não é tocado.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from core.ui import _load_font  # noqa: E402  (reaproveita a busca de fontes)

SIZE = 512
SS = 3  # supersampling: desenha 3x maior e reduz, para bordas suaves

SKIN = (246, 203, 168, 255)
OUTLINE = (48, 40, 56, 255)
CARD_BG = (22, 26, 38, 236)
ACCENTS = [(88, 166, 255), (255, 159, 67), (92, 210, 140), (200, 125, 255), (255, 105, 125)]

PALM = (184, 250, 336, 378)  # x0, y0, x1, y1 (em unidades de 512)
FINGER_X = (203, 241, 279, 317)
FINGER_LEN = (118, 134, 122, 94)  # indicador, médio, anelar, mínimo
FINGER_W = 34
OUT = 4  # espessura do contorno


class Canvas:
    def __init__(self):
        self.img = Image.new("RGBA", (SIZE * SS, SIZE * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    @staticmethod
    def _s(v):
        return v * SS

    def capsule(self, points, width, fill):
        """Linha grossa de pontas arredondadas passando por `points`."""
        w = self._s(width)
        pts = [(self._s(x), self._s(y)) for x, y in points]
        for a, b in zip(pts, pts[1:]):
            self.d.line([a, b], fill=fill, width=int(w))
        for x, y in pts:
            r = w / 2
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def rrect(self, box, radius, fill):
        self.d.rounded_rectangle([self._s(v) for v in box], radius=self._s(radius), fill=fill)

    def shape_pass(self, shapes, outline_first=True):
        """Desenha todos os contornos e depois todos os preenchimentos, de modo que
        o contorno só sobra na silhueta externa e nas frestas entre dedos."""
        for kind, geom, w_or_r in shapes:
            self._draw(kind, geom, w_or_r, OUTLINE, grow=OUT * 2)
        for kind, geom, w_or_r in shapes:
            self._draw(kind, geom, w_or_r, SKIN, grow=0)

    def _draw(self, kind, geom, size, color, grow):
        if kind == "capsule":
            self.capsule(geom, size + grow, color)
        else:
            x0, y0, x1, y1 = geom
            g = grow / 2
            self.rrect((x0 - g, y0 - g, x1 + g, y1 + g), size + g, color)

    def outlined_capsule(self, points, width):
        self.capsule(points, width + OUT * 2, OUTLINE)
        self.capsule(points, width, SKIN)


def draw_hand(c: Canvas, pattern, accent):
    thumb, fingers = pattern[0], list(pattern[1:])
    others_up = sum(1 for f in fingers if f == 1)
    if thumb is None:  # curinga: escolhe a postura mais natural para o desenho
        thumb = 1 if others_up == 4 else 0
    thumbs_up = thumb == 1 and others_up == 0

    # Camada 1: palma, punho e dedos esticados (silhueta única)
    base = [("rrect", PALM, 46), ("rrect", (206, 350, 314, 420), 22)]
    for i, up in enumerate(fingers):
        if up == 1:
            x, top = FINGER_X[i], PALM[1] - FINGER_LEN[i]
            base.append(("capsule", [(x, top + FINGER_W / 2), (x, PALM[1] + 30)], FINGER_W))
    if thumb == 1 and not thumbs_up:
        base.append(("capsule", [(206, 340), (156, 300), (120, 254)], 40))
    c.shape_pass(base)

    # Camada 2: dedos dobrados (aparecem como "nós" sobre a palma)
    for i, up in enumerate(fingers):
        if up != 1:
            x = FINGER_X[i]
            c.outlined_capsule([(x, PALM[1] + 6), (x, PALM[1] + 46)], FINGER_W)

    # Camada 3: polegar dobrado ou "joinha"
    if thumbs_up:
        c.outlined_capsule([(204, 336), (190, 276), (180, 214)], 54)
    elif thumb == 0:
        c.outlined_capsule([(206, 338), (246, 322)], 30)

    # Manga colorida, para "ancorar" a mão visualmente
    c.rrect((190, 404, 330, 452), 16, OUTLINE)
    c.rrect((194, 408, 326, 448), 13, accent + (255,))


def render_card(entry, accent):
    # A mão vai numa camada separada para poder ser centralizada no cartão.
    hand_canvas = Canvas()
    draw_hand(hand_canvas, entry["pattern"], accent)
    hand = hand_canvas.img.resize((SIZE, SIZE), Image.LANCZOS)
    bbox = hand.getbbox()
    hand = hand.crop(bbox)

    card_canvas = Canvas()
    card_canvas.rrect((0, 0, SIZE, SIZE), 44, CARD_BG)
    card = card_canvas.img.resize((SIZE, SIZE), Image.LANCZOS)

    top, bottom = 44, 440  # região útil: abaixo da faixa de acento, acima da legenda
    x = (SIZE - hand.width) // 2
    y = top + (bottom - top - hand.height) // 2
    card.alpha_composite(hand, (x, y))

    d = ImageDraw.Draw(card)
    d.rounded_rectangle([1, 1, SIZE - 2, SIZE - 2], radius=44, outline=(255, 255, 255, 45), width=2)
    d.rounded_rectangle([SIZE // 2 - 34, 20, SIZE // 2 + 34, 27], radius=4, fill=accent + (255,))
    font = _load_font(38, bold=True)
    name = entry["name"]
    l, t, r, b = font.getbbox(name)
    d.text(((SIZE - (r - l)) / 2 - l, 460 - t), name, font=font, fill=(255, 255, 255, 255))
    return card


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Gera as imagens dos gestos de patterns.json")
    p.add_argument("--config", default=os.path.join(BASE_DIR, "patterns.json"))
    p.add_argument("--force", action="store_true", help="sobrescreve imagens que já existem")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        entries = json.load(f)

    created = skipped = 0
    for i, entry in enumerate(entries):
        path = entry["image"]
        path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        if os.path.exists(path) and not args.force:
            print(f"  pulando (já existe): {entry['image']}")
            skipped += 1
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        render_card(entry, ACCENTS[i % len(ACCENTS)]).save(path)
        print(f"  criada: {entry['image']}  ({entry['name']})")
        created += 1
    print(f"Pronto: {created} criada(s), {skipped} mantida(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
