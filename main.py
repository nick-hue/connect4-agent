import pygame 
import math 
import numpy as np
import random
import time
from dataclasses import dataclass
from Classes.Bot import HumanPlayer, BotPlayer

pygame.init()

BOARD_DIMENSIONS = (7,6) # columns, rows
DISPLAY_DIMENSION = 1024, 824
COLUMN_WIDTH = DISPLAY_DIMENSION[0]/7
START_PIXELS_X = 25 
START_PIXELS_Y = 34 
BUFFER_PIXELS_X = 33
BUFFER_PIXELS_Y = 13

DISPLAY = pygame.display.set_mode(DISPLAY_DIMENSION)
CLOCK = pygame.time.Clock()

PIECE_IMAGES = { 1 : pygame.image.load('Images/red_disk.png'), -1 : pygame.image.load('Images/yellow_disk.png')}
PIECE_DIMENSIONS = PIECE_IMAGES[1].get_width(), PIECE_IMAGES[1].get_height()
BOARD_IMAGE = pygame.image.load('Images/board_image.png')

@dataclass
class Piece:
    turn: int
    coords: tuple

class Game():
    def __init__(self) -> None:
        self.BOARD_ARRAY = np.zeros(tuple(reversed(list(BOARD_DIMENSIONS))))
        self.BOARD_ARRAY_CHECK = np.zeros(tuple(reversed(list(BOARD_DIMENSIONS))))
        self.REVERSED_INDS = list(reversed(list(range(6))))
        self.PLACED_PIECES = []

    def render_board(self):
        DISPLAY.blit(BOARD_IMAGE, (0,0))

    def render_pieces(self):
        for piece in self.PLACED_PIECES:
            DISPLAY.blit(PIECE_IMAGES[piece.turn], piece.coords)

    def place_piece(self, turn, col):
        sum_col = sum(self.BOARD_ARRAY[:, col])
    
        y_row = -1

        if sum_col == 0:
            y_row = self.BOARD_ARRAY.shape[0]-1
        else:
            y_row = self.BOARD_ARRAY[:, col].argmax()-1
        
        x_coord = col*PIECE_DIMENSIONS[0] + START_PIXELS_X + col*BUFFER_PIXELS_X
        y_coord = (y_row*PIECE_DIMENSIONS[0]) + START_PIXELS_Y + y_row*BUFFER_PIXELS_Y

        print(x_coord, y_coord)
        piece = Piece(turn=turn, coords=(x_coord,y_coord))
        self.PLACED_PIECES.append(piece)
        self.BOARD_ARRAY[y_row][col]=1
        self.BOARD_ARRAY_CHECK[y_row][col]=turn

    def check_end(self) -> bool:
        return (self.BOARD_ARRAY==1).all()

    def check_win(self, turn) -> bool:
        # Horizontal check
        for row in range(self.BOARD_ARRAY_CHECK.shape[0]):
            for col in range(self.BOARD_ARRAY_CHECK.shape[1] - 3):
                if self.BOARD_ARRAY_CHECK[row][col] == turn and all(self.BOARD_ARRAY_CHECK[row][col+i] == turn for i in range(4)):
                    return True

        # Vertical check
        for col in range(self.BOARD_ARRAY_CHECK.shape[1]):
            for row in range(self.BOARD_ARRAY_CHECK.shape[0] - 3):
                if self.BOARD_ARRAY_CHECK[row][col] == turn and all(self.BOARD_ARRAY_CHECK[row+i][col] == turn for i in range(4)):
                    return True

        # Positive diagonal check
        for col in range(self.BOARD_ARRAY_CHECK.shape[1] - 3):
            for row in range(3, self.BOARD_ARRAY_CHECK.shape[0]):
                if self.BOARD_ARRAY_CHECK[row][col] == turn and all(self.BOARD_ARRAY_CHECK[row-i][col+i] == turn for i in range(4)):
                    return True

        # Negative diagonal check
        for col in range(self.BOARD_ARRAY_CHECK.shape[1] - 3):
            for row in range(self.BOARD_ARRAY_CHECK.shape[0] - 3):
                if self.BOARD_ARRAY_CHECK[row][col] == turn and all(self.BOARD_ARRAY_CHECK[row+i][col+i] == turn for i in range(4)):
                    return True

        if self.check_end():
            return True

        return False

    def get_player_class(self, class_string, name, turn):
        if class_string == 'Human':
            return HumanPlayer(name=name, turn=turn)
        elif class_string == 'Bot':
            return BotPlayer(name=name, turn=turn)

    def main(self):
        running = True
        turn = 1

        player_turn = random.choice((-1,1))

        player_1 = self.get_player_class("Human", "Nikos", player_turn)
        player_2 = self.get_player_class("Bot", "Botakis", -player_turn)
        print(player_1.__class__.__name__)
        print(player_2.__class__.__name__)
            
        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    running = False

                current_player = player_1 if player_1.turn == turn else player_2 # get the current player
                # print(current_player.__class__.__name__)

                if current_player.__class__.__name__ == "HumanPlayer":
                    if event.type==pygame.MOUSEBUTTONDOWN:
                        #print(f"x : {event.pos[0]} , y : {event.pos[1]}")
                        col = math.floor(event.pos[0]/COLUMN_WIDTH)

                        #print("Insert at col ", col)
                        if 0 <= col < 7:
                            if self.BOARD_ARRAY[0, col]==0:
                                self.place_piece(turn, col)
                                print(f"Turn {turn} | Col : {col} | Board \n{self.BOARD_ARRAY_CHECK}")
                                turn = -turn

                elif current_player.__class__.__name__ == "BotPlayer":
                    col = current_player.move(self.BOARD_ARRAY)  
                    self.place_piece(turn, col)
                    turn = -turn
                    # pygame.time.wait()

                else: 
                    print("Invalid player class.")
                    running = False
        
            running = not self.check_win(turn=-turn)
            if not running:
                print(f"{current_player.name} won !")

            DISPLAY.fill('white')
            self.render_board()
            self.render_pieces()
            pygame.display.flip()
            CLOCK.tick(60)


if __name__ == "__main__":
    game = Game()

    game.main()