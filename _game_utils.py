from dataclasses import dataclass
import pygame 
from optionbox import OptionBox

@dataclass
class Piece:
    turn: int
    coords: tuple

@dataclass
class PlayerInput:
    input_rect: pygame.Rect
    player_select: OptionBox
    color: pygame.Color
    active: bool
    player_text: str
    selected: int

def get_font(size):
    return pygame.font.Font("assets/font.ttf", size)