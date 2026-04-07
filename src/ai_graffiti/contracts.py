from dataclasses import dataclass
from typing import Tuple

@dataclass
class GestureState:
    cursor_pos: Tuple[int, int]
    is_painting: bool
    is_erasing: bool
    brush_color: Tuple[int, int, int]  # (B, G, R)
