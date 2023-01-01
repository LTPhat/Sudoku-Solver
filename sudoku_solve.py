import numpy as np


class Sudoku_solver():
    """
    Solve Sudoku using Backtracking algorithm
    """

    def __init__(self, board, size):
        self.board = board
        self.size = size 

    def print_board(self):
        """
        Visualize result board
        """
        for i in range(len(self.board)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - - - ")

        for j in range(len(self.board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(self.board[i][j])
            else:
                print(str(self.board[i][j]) + " ", end="")


    def valid(self, num, pos):
        """
        Check valid board when adding new num in position pos  
        """

        # Check valid row
        for j in range(len(self.board[0])):
            if self.board[pos[0]][j] == num and pos[1] != j:
                return False
        
        # Check valid column
        for i in range(len(self.board)):
            if self.board[i][pos[1]] == num and pos[0] != i:
                return False

        # Check valid box
        # There are 9 boxes

        box_x = pos[0] // 3
        box_y = pos[1] // 3

        for i in range(box_x*3, box_x*3+3):
            for j in range(box_y*3, box_y*3+3):
                if self.board[i][j] == num and (i, j) != pos:
                    return False
        return True


    def find_empty_cell(self):
        """
        Find empty cell and return its position
        """
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    return (i, j)
        return None

    def solve(self):
        pos = self.find_empty_cell()
        # Base case, complete the board
        if not pos:
            return True
        else:
            row, col = pos
        
        for i in range(1, 10):
            if self.valid(i, (row, col)):
                self.board[row][col] = i

                if self.solve():
                    return True

                self.board[row][col] = 0
        return False 