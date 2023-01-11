import pygame
import numpy as np

slow_mode = False
render_pygame = True
action_line_enabled = True
action_line = []
peek_enabled = False

CLOSED = 9
SQUARE_SIZE = 50

pygame.init()

def setup_screen(rows, cols):
    screen = pygame.display.set_mode((rows * SQUARE_SIZE, cols * SQUARE_SIZE))
    return screen
     
     
def input_loop():
    global training
    global slow_mode
    global render_pygame
    global action_line_enabled
    global peek_enabled
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
            training = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            slow_mode = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
            slow_mode = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            render_pygame = not render_pygame
        if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
            action_line_enabled = not action_line_enabled
            render_pygame = not render_pygame
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            peek_enabled = not peek_enabled


def render_loop(state, mine_board, action, screen, rows, cols, reset=False):
    global action_line
    if reset: action_line = []
    
    screen.fill((255,255,255))
    # undo one hot encoding and draw the board
    game_board = np.argmax(state, axis=0)
    
    for y in range(rows):
        for x in range(cols):
            if state[CLOSED, y, x] == 1:
                pygame.draw.rect(screen, (144, 238, 144), (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            elif state[0, y, x] == 1:
                pygame.draw.rect(screen, (255, 255, 255), (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            else:
                pygame.draw.rect(screen, (255, 255, 255), (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                text = str(game_board[y, x].item())
                font = pygame.font.SysFont('Arial', 30)
                text_surface = font.render(text, True, (0, 0, 0))
                screen.blit(text_surface, (x * SQUARE_SIZE + 20, y * SQUARE_SIZE + 10))

            if mine_board[y, x] == True:
                pygame.draw.rect(screen, (255, 0, 0), (x * SQUARE_SIZE + 10, y * SQUARE_SIZE + 10, SQUARE_SIZE - 20, SQUARE_SIZE - 20))

            if y == int(action // rows) and x == int(action % cols):
                pygame.draw.rect(screen, (0, 0, 255), (x * SQUARE_SIZE + 5, y * SQUARE_SIZE + 5, 10, 10))

    # render the grid
    for x in range(rows + 1):
        pygame.draw.line(screen, (0, 0, 0), (x * SQUARE_SIZE, 0), (x * SQUARE_SIZE, cols * SQUARE_SIZE))
    for y in range(cols + 1):
        pygame.draw.line(screen, (0, 0, 0), (0, y * SQUARE_SIZE), (rows * SQUARE_SIZE, y * SQUARE_SIZE))

    if action_line_enabled:
        action_line.append(action)
        for i in range(len(action_line) - 1):
            pygame.draw.line(screen, (0, 0, 255), (int(action_line[i] % cols) * SQUARE_SIZE + 5, int(action_line[i] // cols) * SQUARE_SIZE + 5), (int(action_line[i + 1] % rows) * SQUARE_SIZE + 5, int(action_line[i + 1] // cols) * SQUARE_SIZE + 5))
    
    pygame.display.update()
    

def game_loop(state, mine_board, action, screen, rows, cols, reset):
    input_loop()
    
    if render_pygame: render_loop(state, mine_board, action, screen, rows, cols, reset)

    if slow_mode: pygame.time.delay(200)