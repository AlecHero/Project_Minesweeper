import numpy as np

class Minesweeper():
    def __init__(self, board_shape, mines):
        self.rows, self.cols = board_shape
        self.mines = mines
        
        self.visible_board, self.mine_board = self.create_board()
        self.solved_board = self.solve_board()
        
        self.s_board = self.solved_board.copy()
        self.s_board[self.visible_board] = 9
    

    def create_board(self):
        visible_board = np.ones((self.rows, self.cols), dtype=bool)

        mine_board = np.zeros(self.rows*self.cols, dtype=int) #<-- flat
        mine_board[np.random.choice(self.rows*self.cols, self.mines, replace=False)] = 1
        mine_board = mine_board.reshape(self.rows, self.cols) #<-- grid
        
        return visible_board, mine_board


    def get_slice(self, i, j):
        min_x = max(0, i-1)
        max_x = min(self.rows-1, i+1)
        min_y = max(0, j-1)
        max_y = min(self.cols-1, j+1)
        
        return np.s_[min_x:max_x+1, min_y:max_y+1]


    def solve_board(self):
        solved_board = self.mine_board.copy() * -9
        for i,j in np.argwhere(solved_board == -9):
            solved_board[self.get_slice(i, j)] += 1
        solved_board = np.clip(solved_board, -1, 8)
        return solved_board

    temp_coords_list = []
    def step(self, action, first_move=False):
        x = action // self.rows
        y = action %  self.cols
        done = False
        
        size_0 = np.size(self.temp_coords_list)
        self.temp_coords_list.append((x,y))
        
        self.visible_board[x, y] = False
        
        self.s_board = self.solved_board.copy()
        self.s_board[self.visible_board] = 9
        # self.s_board[x, y] = self.solved_board[x, y]
        
        if self.solved_board[x, y] == 0:
            a = np.argwhere(self.s_board == 0)
            
            self.visible_board[self.get_slice(x, y)] = False
            self.s_board = self.solved_board.copy()
            self.s_board[self.visible_board] = 9
        
            b = np.argwhere(self.s_board == 0)
            
            new_coords = b[len(a):]
        
        

        
        if self.mine_board[x,y] and first_move:
            min_pos = np.argmin(self.mine_board) // 10, np.argmin(self.mine_board) % 10
            self.mine_board[x,y] = 0
            self.mine_board[min_pos] = 1
            self.solved_board = self.solve_board()
        elif self.mine_board[x, y] or np.sum(self.visible_board) == self.mines:
            done = True
        
        return self.visible_board, self.solved_board, done

    
    def render(self, print_board=True):
        str_board = np.array2string(self.s_board, separator="  ")
        str_board = str_board.replace('9', '█')
        str_board = str_board.replace('0', '░')
        str_board = str_board.replace("[", " ")
        str_board = str_board.replace("]", " ")
        
        if print_board: print(str_board)