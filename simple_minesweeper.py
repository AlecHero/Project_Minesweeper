import numpy as np
# import pygame
np.random.seed(1234)
import os

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
                
        
        # uncover_grid(neighbor, grid, grid_visible, neighbors_to_check)
        
        # if grid_visible[neighbor] == "-":
            # grid_visible = uncover_grid(neighbor, grid, grid_visible)


grid_print = lambda grid: print(np.array2string(grid, separator='  ', formatter={'str_kind': lambda x: x if x else ' '}))

if __name__ == "__main__":
    grid = create_grid(15, 15, 50)
    grid_visible = np.full(grid.shape, "-", dtype=str)
    grid_print(grid_visible)
    
    coord_list = [tuple([int(num) for num in input().split(",")])]
    grid = safe_first_click(grid, *coord_list[0])
    
    while True:        
        i = 0
        while i < len(coord_list):
            grid_visible, new_coords = uncover_grid(coord_list[i], grid, grid_visible)
            for coord in new_coords:
                coord_list.append(coord)
            i+=1
            
            os.system("cls" if os.name == "nt" else "clear")
            grid_print(grid_visible)
        
        coord_list = [tuple([int(num) for num in input().split(",")])]
            
        