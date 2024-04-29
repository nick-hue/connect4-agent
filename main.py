import pygame 
from tkinter import *
from tkinter import messagebox
from button import Button

import math 
import numpy as np
import sys
import random

from dataclasses import dataclass
from players.Bot import HumanPlayer, BotPlayer
from players.dqn_agent import DQN
import torch

pygame.init()
scale_factor = 0.75

# GUI VARIABLES
BOARD_DIMENSIONS = (7,6) # columns, rows
DISPLAY_DIMENSION = 1024*scale_factor, 1024*scale_factor
COLUMN_WIDTH = DISPLAY_DIMENSION[0]*scale_factor/7
START_PIXELS_X = 25 * scale_factor
START_PIXELS_Y = 34 * scale_factor
BUFFER_PIXELS_X = 33 * scale_factor
BUFFER_PIXELS_Y = 13 * scale_factor

# DQN INIT VARIABLES
gamma = 0.99
copy_step = 25
hidden_units = [100, 200, 200, 100]
max_experiences = 1000
min_experiences = 100
batch_size = 32
lr = 0.01
epsilon = 0.25   
decay = 0.99
min_epsilon = 0.05
random_episodes = 10000
negamax_episodes = 2000
precision = 7

# PYGAME VARIABLES
DISPLAY = pygame.display.set_mode(DISPLAY_DIMENSION)
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("monospace", 32)
pygame.display.set_caption("Setup Menu")

# START MENU 
MENU_BG_IMAGE_tmp = pygame.image.load('Assets/menu_background_image.png')
MENU_BG_IMAGE = pygame.transform.scale(MENU_BG_IMAGE_tmp, (MENU_BG_IMAGE_tmp.get_width()*scale_factor, MENU_BG_IMAGE_tmp.get_height()*scale_factor))

# CONNECT 4 GAME
RED_DISK_IMAGE = pygame.image.load('Assets/red_disk.png')
YELLOW_DISK_IMAGE = pygame.image.load('Assets/yellow_disk.png')
PIECE_IMAGES = { 1 : pygame.transform.scale(RED_DISK_IMAGE, (RED_DISK_IMAGE.get_width()*scale_factor, RED_DISK_IMAGE.get_height()*scale_factor)),\
                 -1 : pygame.transform.scale(YELLOW_DISK_IMAGE, (YELLOW_DISK_IMAGE.get_width()*scale_factor, YELLOW_DISK_IMAGE.get_height()*scale_factor))}
PIECE_DIMENSIONS = PIECE_IMAGES[1].get_width(), PIECE_IMAGES[1].get_height()
BOARD_IMAGE_tmp = pygame.image.load('Assets/board_image.png')
BOARD_IMAGE = pygame.transform.scale(BOARD_IMAGE_tmp, (BOARD_IMAGE_tmp.get_width()*scale_factor, BOARD_IMAGE_tmp.get_height()*scale_factor))


@dataclass
class Piece:
    turn: int
    coords: tuple

def get_font(size): # Returns Press-Start-2P in the desired size
    return pygame.font.SysFont("monospace", size)


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

    def render_text(self, display_string):
        label = FONT.render(display_string, 1, 'black')
        DISPLAY.blit(label, (100, 900))

    def place_piece(self, turn, col):
        sum_col = sum(self.BOARD_ARRAY[:, col])
    
        y_row = -1

        if sum_col == 0:
            y_row = self.BOARD_ARRAY.shape[0]-1
        else:
            y_row = self.BOARD_ARRAY[:, col].argmax()-1
        
        x_coord = col*PIECE_DIMENSIONS[0] + START_PIXELS_X + col*BUFFER_PIXELS_X
        y_coord = (y_row*PIECE_DIMENSIONS[0]) + START_PIXELS_Y + y_row*BUFFER_PIXELS_Y

        #print(x_coord, y_coord)
        piece = Piece(turn=turn, coords=(x_coord,y_coord))
        self.PLACED_PIECES.append(piece)
        self.BOARD_ARRAY[y_row][col]=1
        self.BOARD_ARRAY_CHECK[y_row][col] = 1 if turn==1 else 2

    def check_end(self) -> bool:
        return (self.BOARD_ARRAY==1).all()

    def check_win(self, turn) -> bool:
        # input turn is 1 or -1
        # Horizontal check
        turn = 1 if turn==1 else 2
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

    def get_player_class(self, class_string, name, turn, number):
        if class_string == 'Human':
            return HumanPlayer(name=name, turn=turn, player_number=number)
        elif class_string == 'Bot':
            return BotPlayer(name=name, turn=turn, player_number=number)
        elif class_string == 'DQNAgent':
            agent = DQN(name, turn, number, 43, 7, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
            agent.load_weights("weights_28_04_2024_12_01_34.pth")
            return agent

    def main(self):
        running = True
        turn = 1

        player_turn = random.choice((-1,1))

        player_1 = self.get_player_class("Human", "Nikos", player_turn, 1)
        player_2 = self.get_player_class("DQNAgent", "Botakis", -player_turn, 2)
        
        playing_string = f"{player_1.__class__.__name__} - {player_1.name} vs {player_2.__class__.__name__} - {player_2.name}"
        print(playing_string)

        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()


                current_player = player_1 if player_1.turn == turn else player_2 # get the current player
 
                if current_player.__class__.__name__ == "HumanPlayer":
                    if event.type==pygame.MOUSEBUTTONDOWN:
                        col = math.floor(event.pos[0]*scale_factor/COLUMN_WIDTH)

                        if 0 <= col < 7:
                            if self.BOARD_ARRAY[0, col]==0:
                                self.place_piece(turn, col)
                                print(f"Turn {turn} | Col : {col} | Board \n{self.BOARD_ARRAY_CHECK}")
                                turn = -turn

                elif current_player.__class__.__name__ == "BotPlayer":
                    col = current_player.move(self.BOARD_ARRAY)  
                    self.place_piece(turn, col)
                    turn = -turn

                elif current_player.__class__.__name__ == "DQN":
                    observation = {'board': self.BOARD_ARRAY_CHECK.flatten(), 'mark': turn}
                    # configuration = {'columns': 7}

                    actions = current_player.model(torch.from_numpy(current_player.preprocess(observation)).float()).detach().numpy()
                    for i in range(len(actions)) :
                        if observation['board'][i] != 0 :
                            actions[i] = -1e7
                    col = int(np.argmax(actions))
                    print(col)
                    self.place_piece(turn, col)
                    turn = -turn

                else: 
                    print("Invalid player class.")
                    running = False
        
            running = not self.check_win(turn=-turn)
            if not running:
                winning_string = f"{current_player.name} wins!"
                print(winning_string)
                self.render_pieces()
                pygame.display.flip()
                CLOCK.tick(60)
                Tk().wm_withdraw() #to hide the main window
                messagebox.showinfo('Game is over',winning_string)
                    
            DISPLAY.fill('white')
            self.render_board()
            self.render_pieces()
            self.render_text(playing_string)
            pygame.display.flip()
            CLOCK.tick(60)

def start_menu():
    while True:
        DISPLAY.blit(MENU_BG_IMAGE, (0, 0))
        
        MENU_MOUSE_POSITION = pygame.mouse.get_pos()

        START_BUTTON = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(DISPLAY_DIMENSION[0]/2, DISPLAY_DIMENSION[0]/4), 
                                text_input="START", font=get_font(75), base_color="#d7fcd4", hovering_color="White")
        
        QUIT_BUTTON = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(DISPLAY_DIMENSION[0]/2, 3*DISPLAY_DIMENSION[0]/4), 
                                text_input="QUIT", font=get_font(75), base_color="#d7fcd4", hovering_color="White")
        
        START_BUTTON.changeColor(MENU_MOUSE_POSITION)
        START_BUTTON.update(DISPLAY)

        QUIT_BUTTON.changeColor(MENU_MOUSE_POSITION)
        QUIT_BUTTON.update(DISPLAY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if START_BUTTON.checkForInput(MENU_MOUSE_POSITION):
                    game = Game()
                    game.main()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POSITION):
                    pygame.quit()
                    sys.exit()
                    
        pygame.display.update()


if __name__ == "__main__":
    start_menu()