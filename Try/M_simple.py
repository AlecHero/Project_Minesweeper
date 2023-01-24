import numpy as np

class Minesweeper():
    def __init__(self, board_shape, mines, rewards):
        self.rows, self.cols = board_shape
        self.mines = mines
        self.reward_win, self.reward_loss, self.reward_step = rewards
        
        self.board, self.mine_board = self.create_board()
        self.solved_board = self.solve_board()
        

    def create_board(self):
        board = np.zeros((10, self.rows, self.cols), dtype=int)
        board[-1] = 1
        
        mine_board = np.zeros(self.rows*self.cols, dtype=int) #<-- flat
        mine_board[np.random.choice(self.rows*self.cols, self.mines, replace=False)] = 1
        mine_board = mine_board.reshape(self.rows, self.cols) #<-- grid
        
        return board, mine_board

    
    def solve_board(self):
        solved_board = self.mine_board.copy() * -9
        for i,j in np.argwhere(solved_board == -9):
            min_x = max(0, i-1)
            max_x = min(self.rows-1, i+1)
            min_y = max(0, j-1)
            max_y = min(self.cols-1, j+1)
            
            solved_board[min_x:max_x+1, min_y:max_y+1] += 1
        solved_board = np.clip(solved_board, -1, 8)
        return solved_board

    def bob(self, new_board, mine_board, x, y):
        new_board = self.board.copy()
        new_board[-1, x, y] = False

        mines = self.solved_board[x, y]
        new_board[mines, x, y] = True

        if mines == 0:
            for _x in range(max(0, x-1), min(self.rows, x+2)):
                for _y in range(max(0, y-1), min(self.cols, y+2)):
                    if new_board[-1, _x, _y]:
                        mines = self.solved_board[_x, _y]
                        if mines == 0:
                            new_board = self.bob(new_board, mine_board, _x, _y)
                        else:
                            new_board[mines, _x, _y] = True
                            new_board[-1, _x, _y] = False
    

    def step(self, action, first_move=False):
        x = action // self.rows
        y = action %  self.cols
        done = False
        
        # If first move is bomb
        if self.mine_board[x,y] and first_move:
            reward = self.reward_step
            
            min_pos = np.argmin(self.mine_board) // 10, np.argmin(self.mine_board) % 10
            self.mine_board[x,y] = 0
            self.mine_board[min_pos] = 1

            self.solved_board = self.solve_board()
        # If move is bomb
        elif self.mine_board[x, y]:
            reward = self.reward_loss
            self.board[-1, x, y] = 0
            done = True
        # If move is safe
        else:
            reward = self.reward_step
            self.board[-1, x, y] = 0
            self.board[self.solved_board[x, y], x, y] = 1
        
        # Check if game is over
        if np.sum(self.board[-1]) == self.mines:
            reward = self.reward_win
            done = True
        return self.board, self.solved_board, reward, done