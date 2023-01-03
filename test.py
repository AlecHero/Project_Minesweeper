import numpy as np
# np.random.seed(1234)
import pygame

def update_grid(grid):
    x_dim, y_dim = grid.shape
    grid[grid != -1] = 0 # Reset all non-bomb squares to 0
    
    # Loop through all squares and add 1 for all adjacent bombs
    for i in range(x_dim):
        for j in range(y_dim):
            if grid[i][j] != -1:
                # Add 1 to all adjacent squares:
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if i + k >= 0 and i + k < x_dim and j + l >= 0 and j + l < y_dim:
                            if grid[i + k][j + l] == -1:
                                grid[i][j] += 1
    return grid


def create_grid(x_dim, y_dim, mines):
    # Create a grid of size x by y filled with zeroes
    grid = np.zeros((x_dim, y_dim), dtype=int)

    # Place z # bombs randomly on the grid
    for _ in range(mines):
        x_coord = np.random.randint(0, x_dim)
        y_coord = np.random.randint(0, y_dim)
        grid[x_coord][y_coord] = -1
    
    update_grid(grid)
    
    return grid


def safe_first_click(grid, x, y):
    if grid[x][y] == -1:
        grid[x][y] = 0
        
        for i in range(grid.shape[0]):
            if grid[i, 0] != -1:
                grid[i, 0] = -1
                break

    return update_grid(grid)


def uncover_grid(input_coord, grid, grid_visible):
    neighbors_to_check = []
    
    if grid[input_coord] == -1:
        grid_visible[input_coord] = "X"
        return grid_visible, []
    elif grid[input_coord] != 0:
        grid_visible[input_coord] = str(grid[input_coord])
        return grid_visible, []
    elif grid[input_coord] == 0:
        grid_visible[input_coord] = " "
        cond = lambda x, y: x >= 0 and x < grid.shape[0] and y >= 0 and y < grid.shape[1]
        neighbors = [(input_coord[0]+dx, input_coord[1]+dy) for dx in range(-1, 2) for dy in range(-1, 2) if cond(input_coord[0]+dx, input_coord[1]+dy)]
        
        for coord in neighbors:
            if grid_visible[coord] == "-":
                grid_visible[coord] = "o"
                neighbors_to_check.append(coord)
        return grid_visible, neighbors_to_check

# Define some colors
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


grid = [
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','2','1','1','1','2','2','2','-','-','-','-','-','-','-'],
    ['-','1',',',',',',',',',',','2','-','-','-','-','-','-','-'],
    ['1','1',',','1','1','1',',','1','-','-','-','-','-','-','-'],
    [',',',',',','1','-','1',',','1','-','-','-','-','-','-','-'],
    ['1','1','1','2','-','3','1','2','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']]

def get_grid(coord, is_first, grid, grid_visible):
    coord_list = [coord]
    
    if is_first:
        grid = safe_first_click(grid, *coord_list[0])
    
    i = 0
    while i < len(coord_list):
        grid_visible, new_coords = uncover_grid(coord_list[i], grid, grid_visible)
        for coord in new_coords:
            coord_list.append(coord)
        i+=1

    return grid_visible

# This function will draw the grid on the screen
def draw_grid(screen, grid):
    # Determine the width and height of each cell in the grid
    cell_width = screen.get_width() // len(grid[0])
    cell_height = screen.get_height() // len(grid)

    # Iterate over the cells in the grid
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            # Determine the position and size of the cell
            x = col * cell_width
            y = row * cell_height
            width = cell_width
            height = cell_height

            # Draw the cell
            if grid[row][col] == " ":
                # Draw a white square for an empty cell
                pygame.draw.rect(screen, WHITE, (x, y, width, height))
            elif grid[row][col] == "-":
                # Draw a gray square for an unknown cell
                pygame.draw.rect(screen, GRAY, (x, y, width, height))
                pygame.draw.rect(screen, BLACK, (x, y, width, height), width=1)
            elif grid[row][col] == "X":
                # Draw a red square for a bomb cell
                pygame.draw.rect(screen, RED, (x, y, width, height))
            else:
                # Draw a white square for a numbered cell, and display the number
                pygame.draw.rect(screen, WHITE, (x, y, width, height))
                font = pygame.font.Font(None, 36)
                text = font.render(str(grid[row][col]), True, BLACK)
                screen.blit(text, (x + width // 2 - text.get_width() // 2, y + height // 2 - text.get_height() // 2))
                
                # pygame.draw.rect(screen, BLACK, (x, y, width, height), width=1)

# This function will be called when a button is clicked
def on_button_clicked(pos):
    # Calculate the row and column of the clicked button
    col = pos[0] // (screen.get_width() // len(grid[0]))
    row = pos[1] // (screen.get_height() // len(grid))
    return (row, col)

# Set up Pygame
pygame.init()

# Set the window size and title
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))

grid = create_grid(15, 15, 50)

first = True
try:
    # Main loop of the program
    running = True
    grid_visible = np.full(grid.shape, "-", dtype=str)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONUP:
                grid_visible = get_grid(on_button_clicked(event.pos), first, grid, grid_visible)
                first = False
        screen.fill(BLACK)
        draw_grid(screen, grid_visible)
        pygame.display.flip()
except Exception as e:
    print(f"An error occurred: {e}")