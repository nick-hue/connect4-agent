import pygame 
from tkinter import *
from tkinter import messagebox
from button import Button
from _game_utils import Piece, PlayerInput,get_font
from optionbox import OptionBox
import math 
import numpy as np
import sys
import random
from test_sub import my_agent
from players.Bot import HumanPlayer, BotPlayer
from players.dqn_agent import DQN
import torch

pygame.init()

# GUI VARIABLES
scale_factor = 0.8125 # (13/16)
BOARD_DIMENSIONS = (7,6) # columns, rows
DISPLAY_DIMENSION = 1024*scale_factor, 1024*scale_factor
COLUMN_WIDTH = DISPLAY_DIMENSION[0]*scale_factor/7
START_PIXELS_X = 25 * scale_factor
START_PIXELS_Y = 34 * scale_factor
BUFFER_PIXELS_X = 33 * scale_factor
BUFFER_PIXELS_Y = 13 * scale_factor

# OPTION TYPES
PLAYER_TYPES = ["Human", "Random Bot", "Agent"]

# PYGAME VARIABLES
DISPLAY = pygame.display.set_mode(DISPLAY_DIMENSION)
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("monospace", 32)
color_black = (0,0,0)
color_grey = (150, 150, 150)
COLOR_ACTIVE = pygame.Color('lightskyblue3')
COLOR_INACTIVE = pygame.Color('gray36')

# START MENU 
MENU_BG_IMAGE_tmp = pygame.image.load('Assets/menu_background_image.png')
MENU_BG_IMAGE = pygame.transform.scale(MENU_BG_IMAGE_tmp, (MENU_BG_IMAGE_tmp.get_width()*scale_factor, MENU_BG_IMAGE_tmp.get_height()*scale_factor))
pygame.display.set_caption("Setup Menu")

# CONNECT 4 GAME
RED_DISK_IMAGE = pygame.image.load('Assets/red_disk.png')
YELLOW_DISK_IMAGE = pygame.image.load('Assets/yellow_disk.png')
PIECE_IMAGES = { 1 : pygame.transform.scale(RED_DISK_IMAGE, (RED_DISK_IMAGE.get_width()*scale_factor, RED_DISK_IMAGE.get_height()*scale_factor)),\
                 -1 : pygame.transform.scale(YELLOW_DISK_IMAGE, (YELLOW_DISK_IMAGE.get_width()*scale_factor, YELLOW_DISK_IMAGE.get_height()*scale_factor))}
PIECE_DIMENSIONS = PIECE_IMAGES[1].get_width(), PIECE_IMAGES[1].get_height()
BOARD_IMAGE_tmp = pygame.image.load('Assets/board_image.png')
BOARD_IMAGE = pygame.transform.scale(BOARD_IMAGE_tmp, (BOARD_IMAGE_tmp.get_width()*scale_factor, BOARD_IMAGE_tmp.get_height()*scale_factor))


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
        elif class_string == 'Random Bot':
            return BotPlayer(name=name, turn=turn, player_number=number)
        elif class_string == 'Agent':
            agent = DQN(name, turn, number)
            agent.load_weights("weights_01_05_2024_21_36_36__final.pth")
            #agent.load_optimizer("test.pth")
            return agent

    def main(self, p1, p2):
        running = True
        turn = 1
        player_turn = random.choice((-1,1))

        player_1 = self.get_player_class("Human", "Nikos", player_turn, 1)
        player_2 = self.get_player_class("Agent", "Botakis", -player_turn, 2)
        
        # player_1 = self.get_player_class(PLAYER_TYPES[p1.selected], p1.player_text, player_turn, 1)
        # player_2 = self.get_player_class(PLAYER_TYPES[p2.selected], p2.player_text, -player_turn, 2)

        playing_string = f"{p1.player_text}-{PLAYER_TYPES[p1.selected]} VS {p2.player_text}-{PLAYER_TYPES[p2.selected]}"
        print(playing_string)

        VERSUS_TEXT = get_font(32).render(playing_string, True, color_black)
        VERSUS_RECT = VERSUS_TEXT.get_rect(center=(DISPLAY_DIMENSION[0]/2, 7*DISPLAY_DIMENSION[0]/8))
        DISPLAY.blit(VERSUS_TEXT, VERSUS_RECT)

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
                    configuration = {'columns':7}

                    #print(self.BOARD_ARRAY_CHECK.flatten())
                    # actions = current_player.model(torch.from_numpy(current_player.preprocess(observation)).float()).detach().numpy()
                    # actions = current_player.predict(np.atleast_2d(current_player.preprocess(observation)))[0].detach().numpy()
                    # # print(current_player.optimizer.state_dict())
                    # #

                    # print(actions)
                    col = my_agent(observation=observation, configuration=configuration)
                    print(col)
                    # for i in range(len(actions)) :
                    #     if observation['board'][i] != 0 :
                    #         actions[i] = -1e7
                    # col = int(np.argmax(actions))
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
            DISPLAY.blit(VERSUS_TEXT, VERSUS_RECT)
            pygame.display.flip()
            CLOCK.tick(60)



def start_menu():
    player_1_input = PlayerInput(input_rect=pygame.Rect(DISPLAY_DIMENSION[0]/8, 3*DISPLAY_DIMENSION[0]/8+100, 250, 55), \
                     player_select=OptionBox(DISPLAY_DIMENSION[0]/8, 3*DISPLAY_DIMENSION[0]/8+100+55, 250, 55, color_grey, (100, 200, 255), get_font(30), PLAYER_TYPES), \
                     color=COLOR_INACTIVE,\
                     active=False,\
                     player_text='Player1',
                     selected=0)
    player_2_input = PlayerInput(input_rect=pygame.Rect(5*DISPLAY_DIMENSION[0]/8-50, 3*DISPLAY_DIMENSION[0]/8+100, 250, 55), \
                     player_select=OptionBox(5*DISPLAY_DIMENSION[0]/8-50, 3*DISPLAY_DIMENSION[0]/8+100+55, 250, 55, color_grey, (100, 200, 255), get_font(30), PLAYER_TYPES), \
                     color=COLOR_INACTIVE,\
                     active=False,\
                     player_text='Player2',
                     selected=0)

    while True:
        DISPLAY.blit(MENU_BG_IMAGE, (0, 0))
        
        MENU_MOUSE_POSITION = pygame.mouse.get_pos()

        MENU_TEXT = get_font(100).render("CONNECT 4", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(DISPLAY_DIMENSION[0]/2, DISPLAY_DIMENSION[0]/8))

        START_BUTTON = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(DISPLAY_DIMENSION[0]/2, 3*DISPLAY_DIMENSION[0]/8), 
                                text_input="START", font=get_font(75), base_color="#d7fcd4", hovering_color="White")
        
        QUIT_BUTTON = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(DISPLAY_DIMENSION[0]/2, 7*DISPLAY_DIMENSION[0]/8), 
                                text_input="QUIT", font=get_font(75), base_color="#d7fcd4", hovering_color="White")
        
        DISPLAY.blit(MENU_TEXT, MENU_RECT)

        START_BUTTON.changeColor(MENU_MOUSE_POSITION)
        START_BUTTON.update(DISPLAY)

        text_surface_p1 = get_font(35).render(player_1_input.player_text, True, color_black)
        pygame.draw.rect(DISPLAY, player_1_input.color, player_1_input.input_rect)
        DISPLAY.blit(text_surface_p1, (player_1_input.input_rect.x+5, player_1_input.input_rect.y+5))
        player_1_input.player_select.draw(DISPLAY)
        player_1_input.input_rect.width = max(text_surface_p1.get_width() + 10, 250)

        vs_text = get_font(35).render("Vs", True, color_black)
        x_vs_text = (player_1_input.input_rect.x + player_1_input.input_rect.w + player_2_input.input_rect.x)/2 - 25
        y_vs_text = player_1_input.input_rect.y + 25
        DISPLAY.blit(vs_text, (x_vs_text,y_vs_text))

        text_surface_p2 = get_font(35).render(player_2_input.player_text, True, color_black)
        pygame.draw.rect(DISPLAY, player_2_input.color, player_2_input.input_rect)
        DISPLAY.blit(text_surface_p2, (player_2_input.input_rect.x+5, player_2_input.input_rect.y+5))
        player_2_input.player_select.draw(DISPLAY)
        player_2_input.input_rect.width = max(text_surface_p2.get_width() + 10, 250)

        QUIT_BUTTON.changeColor(MENU_MOUSE_POSITION)
        QUIT_BUTTON.update(DISPLAY)

        event_list = pygame.event.get()
        for event in event_list:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if START_BUTTON.checkForInput(MENU_MOUSE_POSITION):
                    game = Game()
                    print(f"{player_1_input.selected}, {player_2_input.selected}")
                    game.main(player_1_input, player_2_input)
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POSITION):
                    pygame.quit()
                    sys.exit()

                if player_1_input.input_rect.collidepoint(event.pos):
                    player_1_input.active = True
                    player_1_input.color = COLOR_ACTIVE
                else:
                    player_1_input.active = False
                    player_1_input.color = COLOR_INACTIVE
                
                if player_2_input.input_rect.collidepoint(event.pos):
                    player_2_input.active = True
                    player_2_input.color = COLOR_ACTIVE
                else:
                    player_2_input.active = False
                    player_2_input.color = COLOR_INACTIVE

            if event.type == pygame.KEYDOWN and player_1_input.active:
                if event.key == pygame.K_BACKSPACE:
                    player_1_input.player_text = player_1_input.player_text[:-1]
                else:
                    player_1_input.player_text += event.unicode

            if event.type == pygame.KEYDOWN and player_2_input.active:
                if event.key == pygame.K_BACKSPACE:
                    player_2_input.player_text = player_2_input.player_text[:-1]
                else:
                    player_2_input.player_text += event.unicode

        tmp = player_1_input.player_select.update(event_list) 
        if tmp >= 0:
            player_1_input.selected = tmp

        tmp = player_2_input.player_select.update(event_list)
        if tmp >= 0:
            player_2_input.selected = tmp
        
        pygame.display.update()


if __name__ == "__main__":
    start_menu()