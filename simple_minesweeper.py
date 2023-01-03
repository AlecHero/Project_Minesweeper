import numpy as np


def create_grid(x_dim, y_dim, mines):
    # Create a grid of size x by y filled with zeroes
    grid = np.zeros((x_dim, y_dim), dtype=int)

    # Place z # bombs randomly on the grid
    for _ in range(mines):
        x_coord = np.random.randint(0, x_dim)
        y_coord = np.random.randint(0, y_dim)
        grid[x_coord][y_coord] = -1
    
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



def main():
    print(create_grid(10,10,20))



if __name__ == "__main__":
    main()