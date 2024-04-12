import pygame 
import math 
import numpy as np
from dataclasses import dataclass

pygame.init()

BOARD_DIMENSIONS = (7,6) # columns, rows
DISPLAY_DIMENSION = 1024, 824
START_PIXELS_X = 25 
START_PIXELS_Y = 34 
BUFFER_PIXELS_X = 33
BUFFER_PIXELS_Y = 13

BUFFER_PIXELS = 40                #   PIXELS TO BUFFER BETWEEN COLUMN 

DISPLAY = pygame.display.set_mode(DISPLAY_DIMENSION)
CLOCK = pygame.time.Clock()

PIECE_IMAGES = { 1 : pygame.image.load('Images/red_disk.png'), -1 : pygame.image.load('Images/yellow_disk.png')}
PIECE_DIMENSIONS = PIECE_IMAGES[1].get_width(), PIECE_IMAGES[1].get_height()
BOARD_IMAGE = pygame.image.load('Images/board_image.png')

BOARD_ARRAY = np.zeros(tuple(reversed(list(BOARD_DIMENSIONS))))
REVERSED_INDS = list(reversed(list(range(6))))
PLACED_PIECES = []

@dataclass
class Piece:
    turn: int
    coords: tuple

def render_board():
    DISPLAY.blit(BOARD_IMAGE, (0,0))

def render_pieces():
    for piece in PLACED_PIECES:
        DISPLAY.blit(PIECE_IMAGES[piece.turn], piece.coords)

def place_piece(turn, col):
    sum_col = sum(BOARD_ARRAY[:, col])
    print("sum col", sum_col)

    if sum_col == 0:
        print("here")
        y_row = BOARD_ARRAY.shape[0]-1
    else:
        y_row = BOARD_ARRAY[:, col].argmax()-1
    
    x_coord = col*PIECE_DIMENSIONS[0] + START_PIXELS_X + col*BUFFER_PIXELS_X
    y_coord = (y_row*PIECE_DIMENSIONS[0]) + START_PIXELS_Y + y_row*BUFFER_PIXELS_Y

    print(x_coord, y_coord)
    piece = Piece(turn=turn, coords=(x_coord,y_coord))
    PLACED_PIECES.append(piece)
    BOARD_ARRAY[y_row][col]=1


running = True
turn = 1

while running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            running = False

        if event.type==pygame.MOUSEBUTTONDOWN:
            print(f"x : {event.pos[0]} , y : {event.pos[1]}")
            col = math.floor((event.pos[0]-START_PIXELS_X)/PIECE_DIMENSIONS[0])

            print("Insert at col ", col)
            if 0 <= col < 7:
                if BOARD_ARRAY[0, col]==0:
                    place_piece(turn, col)
                    print(f"Turn {turn} | Col : {col} | Board \n{BOARD_ARRAY}")
                    turn = -turn
        

    DISPLAY.fill('white')
    render_board()
    render_pieces()
    pygame.display.flip()
    CLOCK.tick(60)

