"""Compara os dedos esticados com patterns.json e estabiliza o resultado no tempo."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

Pattern = Tuple[Optional[int], ...]
Entry = Dict[str, object]

FINGER_COUNT = 5


class PatternMatcher:
    def __init__(self, config_path: str, stability_frames: int = 5, release_frames: int = 4):
        """
        config_path: JSON com uma lista de gestos:
            [{"name": "Paz", "pattern": [null, 1, 1, 0, 0], "image": "assets/paz.png"}, ...]
            No padrão, 1 = dedo esticado, 0 = dobrado e null = tanto faz (curinga).
        stability_frames: quadros seguidos com o MESMO gesto para ele ser confirmado.
            Evita falsos positivos quando a mão só passa por uma pose no caminho.
        release_frames: quadros seguidos SEM esse gesto para ele ser dispensado.
            É propositalmente curto (a imagem some logo quando a mão sai), mas maior
            que 1, para que uma falha isolada da detecção não faça a imagem piscar.
        """
        self._entries = self._load_config(config_path)
        self._stability_frames = max(1, stability_frames)
        self._release_frames = max(1, release_frames)
        self.reset()

    def reset(self) -> None:
        self._active_entry: Optional[Entry] = None
        self._candidate_name: Optional[str] = None
        self._candidate_count = 0

    @property
    def entries(self) -> List[Entry]:
        return list(self._entries)

    @staticmethod
    def _parse_pattern(name: str, raw) -> Pattern:
        if not isinstance(raw, (list, tuple)) or len(raw) != FINGER_COUNT:
            raise ValueError(
                f"Gesto '{name}': 'pattern' precisa ter {FINGER_COUNT} valores "
                f"[polegar, indicador, médio, anelar, mínimo], recebi {raw!r}"
            )
        parsed: List[Optional[int]] = []
        for value in raw:
            if value is None:
                parsed.append(None)
            elif value in (0, 1) and not isinstance(value, bool):
                parsed.append(int(value))
            else:
                raise ValueError(f"Gesto '{name}': valores válidos são 0, 1 ou null, recebi {value!r}")
        return tuple(parsed)

    @classmethod
    def _load_config(cls, config_path: str) -> List[Entry]:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Arquivo de gestos não encontrado: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("patterns.json deve conter uma lista de gestos")

        entries: List[Entry] = []
        seen: Dict[Pattern, str] = {}
        for item in data:
            name = item["name"]
            pattern = cls._parse_pattern(name, item["pattern"])
            if pattern in seen:
                raise ValueError(
                    f"Os gestos '{seen[pattern]}' e '{name}' têm o mesmo padrão {list(pattern)}; "
                    "o segundo nunca seria reconhecido."
                )
            seen[pattern] = name
            entries.append({"name": name, "pattern": pattern, "image": item["image"]})
        return entries

    def _find_entry(self, fingers: Optional[Sequence[int]]) -> Optional[Entry]:
        """Gesto que combina com os dedos. Se vários combinam (por causa de curingas),
        vence o mais específico, ou seja, o que tem menos curingas."""
        if fingers is None:
            return None
        key = tuple(fingers)
        best: Optional[Entry] = None
        best_specificity = -1
        for entry in self._entries:
            pattern: Pattern = entry["pattern"]  # type: ignore[assignment]
            if all(p is None or p == f for p, f in zip(pattern, key)):
                specificity = sum(p is not None for p in pattern)
                if specificity > best_specificity:
                    best, best_specificity = entry, specificity
        return best

    def update(self, fingers: Optional[Sequence[int]]) -> Optional[Entry]:
        """Recebe a leitura de UM quadro. Devolve o gesto ativo (estável) ou None."""
        raw = self._find_entry(fingers)
        raw_name = raw["name"] if raw else None
        active_name = self._active_entry["name"] if self._active_entry else None

        if raw_name == active_name:
            # Leitura concorda com o que já está ativo: nada pendente.
            self._candidate_name, self._candidate_count = None, 0
            return self._active_entry

        if raw_name == self._candidate_name:
            self._candidate_count += 1
        else:
            self._candidate_name, self._candidate_count = raw_name, 1

        needed = self._stability_frames if raw is not None else self._release_frames
        if self._candidate_count >= needed:
            self._active_entry = raw
            self._candidate_name, self._candidate_count = None, 0
        return self._active_entry
