import random
import copy


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    # Helper method to check if is in drop phase
    def check_drop(self, state):
        count = 0
        #Traverse through each row and column
        for row in range(len(state)):
            for col in range(len(state[row])):
                # Check if it's red or black
                if state[row][col] == 'r' or state[row][col] == 'b':
                    count += 1 # Increment count
        
        # Check if count is greater than or equal to 8
        if (count >= 8):
            return False # False if it is
        else:
            return True # True if it isn't

    # Return a list of successors that takes in board state
    def succ(self, state, marker):
        # List of successors
        succ = []

        # Check if it's in drop phase
        drop_phase = self.check_drop(state)

        if drop_phase:
            # Traverse through every row and col
            for row in range(len(state)):
                for col in range(len(state[row])):
                    # Check if it's empty
                    if state[row][col] == ' ':
                        # Copy the state and append
                        temp = copy.deepcopy(state)
                        temp[row][col] = marker
                        succ.append(temp)
        else:
            # Traverse through every row and col
            for row in range(len(state)):
                for col in range(len(state[row])):
                    # Move markers
                    if state[row][col] == marker:
                        # Down
                        temp = copy.deepcopy(state)
                        if row + 1 < len(temp) and temp[row + 1][col] == ' ':
                            temp[row][col] = ' '
                            temp[row + 1][col] = marker
                            succ.append(temp)

                        # Up
                        temp = copy.deepcopy(state)
                        if row - 1 >= 0 and temp[row - 1][col] == ' ':
                            temp[row][col] = ' '
                            temp[row - 1][col] = marker
                            succ.append(temp)

                        # Right
                        temp = copy.deepcopy(state)
                        if col + 1 < len(temp) and temp[row][col + 1] == ' ':
                            temp[row][col] = ' '
                            temp[row][col + 1] = marker
                            succ.append(temp)

                        # Left
                        temp = copy.deepcopy(state)
                        if col - 1 >= 0 and temp[row][col - 1] == ' ':
                            temp[row][col] = ' '
                            temp[row][col - 1] = marker
                            succ.append(temp)

                        # BL
                        temp = copy.deepcopy(state)
                        if row + 1 < len(temp) and col - 1 >= 0 and temp[row + 1][col - 1] == ' ':
                            temp[row][col] = ' '
                            temp[row + 1][col - 1] = marker
                            succ.append(temp)

                        # BR
                        temp = copy.deepcopy(state)
                        if row + 1 < len(temp) and col + 1 < len(temp) and temp[row + 1][col + 1] == ' ':
                            temp[row][col] = ' '
                            temp[row + 1][col + 1] = marker
                            succ.append(temp)

                        # TL
                        temp = copy.deepcopy(state)
                        if row - 1 >= 0 and col - 1 >= 0 and temp[row - 1][col - 1] == ' ':
                            temp[row][col] = ' '
                            temp[row - 1][col - 1] = marker
                            succ.append(temp)

                        # TR
                        temp = copy.deepcopy(state)
                        if row - 1 >= 0 and col + 1 < len(temp) and temp[row - 1][col + 1] == ' ':
                            temp[row][col] = ' '
                            temp[row - 1][col + 1] = marker
                            succ.append(temp)

        return succ

    # Helper method to calculate heuristic value for horizontal
    def cal_hor(self, state, marker):
        max = 0
        count = 0
        
        # Traverse through every row and col
        for row in range(len(state)):
            jump = False
            for col in range(len(state) - 1):
                # Check if it's equal to a marker
                if state[row][col] == marker:
                    count += 1 # Increment count
                    n = 0

                    # While spot to the right is equal
                    while state[row][col] == state[row][col + 1]:
                        count += 1 # Increment count 
                        col += 1 # Go to next column
                        n += 1

                        # Check if n is greater than or equal to 3
                        if n >= 3:
                            return count / 4
                        
                        # Check if column is out of bounds
                        if col >= len(state) - 1:
                            jump = True
                            break
                
                # Check if max is less than count
                if max < count:
                    max = count
                count = 0

                # Check if jump is true or not
                if jump:
                    break

        return max / 4

    # Helper method to calculate heuristic value for vertical
    def cal_ver(self, state, marker):
        max = 0
        count = 0

        # Traverse through every row and col
        for col in range(len(state)):
            jump = False
            for row in range(len(state) - 1):
                # Check if there is a marker
                if state[row][col] == marker:
                    count += 1
                    n = 0

                    # While the next row is equal
                    while state[row][col] == state[row + 1][col]:
                        count += 1 # Increment count
                        row += 1 # Increment row
                        n += 1

                        # Check if n is greater than or equal to 3
                        if n >= 3:
                            return count / 4
                        
                        # Check if row is out of bounds
                        if row >= len(state) - 1:
                            jump = True
                            break

                # Check if max is less than count
                if max < count:
                    max = count
                count = 0

                # Check if jump is true or not
                if jump:
                    break

        return max / 4

    # Helper method to calculate heuristic value for diagonal /
    def cal_dia_right(self, state, marker):
        max = 0
        count = 0

        # Traverse through every row and col
        for row in range(len(state) - 1):
            jump = False
            for col in range(len(state) - 1):
                # Check if there is a marker
                if state[row][col] == marker:
                    count += 1 # Increment count
                    n = 0

                    # While the BR is equal
                    while state[row][col] == state[row + 1][col + 1]:
                        count += 1 # Increment count
                        row += 1 # Increment row
                        col += 1 # Increment column
                        n += 1

                        # Check if n is greater than or equal to 3
                        if n >= 3:
                            return count / 4
                        
                        # Check if col or row is out of bounds
                        if col >= len(state) - 1 or row >= len(state) - 1:
                            jump = True
                            break

                # Check if max is less than count
                if max < count:
                    max = count
                count = 0

                # Check if jump is true or nto
                if jump:
                    break

        return max / 4

    # Helper method to calculate heuristic value for diagonal \
    def cal_dia_left(self, state, marker):
        max = 0
        count = 0
        for i in range(2, len(state)):
            jump = False
            for j in range(len(state) - 1):
                if state[i][j] == marker:
                    count = count + 1
                    n = 0
                    while state[i][j] == state[i - 1][j + 1]:
                        count = count + 1
                        i = i - 1
                        j = j + 1
                        n = n + 1
                        if n >= 3:
                            return count / 4
                        if i < 0 or j >= len(state) - 1:
                            jump = True
                            break
                if max < count:
                    max = count
                count = 0
                if jump:
                    break
        return max / 4

    # Helper method to calculate heuristic value for square
    def cal_square(self, state, marker):
        max = 0
        count = 0

        # Traverse through the first 4 rows and cols
        for row in range(len(state) - 1):
            for col in range(len(state) - 1):
                # Check if there is a marker and increment if there is at every spot
                if state[row][col] == marker:
                    count += 1 
                if state[row][col + 1] == marker:
                    count += 1
                if state[row + 1][col] == marker:
                    count += 1
                if state[row + 1][col + 1] == marker:
                    count += 1
                
                # Check if max is less than count
                if max < count:
                    max = count

                count = 0

        return max / 4

    # Calculate heuristic value for current state
    def heuristic_game_value(self, state):
        # check if is terminal state
        terminal = self.game_value(state)
        if terminal == 1 or terminal == -1:
            return terminal, state

        my = []
        opp = []

        # heuristic of my
        my.append(self.cal_hor(state, self.my_piece))
        my.append(self.cal_ver(state, self.my_piece))
        my.append(self.cal_dia_left(state, self.my_piece))
        my.append(self.cal_dia_right(state, self.my_piece))
        my.append(self.cal_square(state, self.my_piece))

        # heuristic of opp
        opp.append(self.cal_hor(state, self.opp))
        opp.append(self.cal_ver(state, self.opp))
        opp.append(self.cal_dia_left(state, self.opp))
        opp.append(self.cal_dia_right(state, self.opp))
        opp.append(self.cal_square(state, self.opp))

        max_score = max(my) # Get max score (player)
        min_score = max(opp) # Get min score (AI)

        return max_score + (-1) * min_score, state

    # Find the max heuristic value in the given depth, which is 2 here
    # Return if it is a terminal state
    def Max_Value(self, state, depth):
        max_state = copy.deepcopy(state) # Copy state

        # Check if the game value is not 0
        if self.game_value(state) != 0:
            return self.game_value(state), state

        # Check if depth is greater than 2
        if depth > 2:
            return self.heuristic_game_value(state)
        else:
            a = float('-Inf')

            # Loop through all possible states
            for s in self.succ(state, self.my_piece):
                # Check if the game_value is not 0
                if self.game_value(s) != 0:
                    return self.game_value(s), s
                
                val, curr = self.Min_Value(s, depth + 1) # Get min value

                # Check if min value is greater than a
                if val > a:
                    a = val
                    max_state = s

        return a, max_state

    # Find the min heuristic value in the given depth, which is 2 here
    # Return if it is a terminal state
    def Min_Value(self, state, depth):
        min_state = copy.deepcopy(state) # Copy state

        # Check if game_value is not 0
        if self.game_value(state) != 0:
            return self.game_value(state), state

        # Check if depth is greater than 2
        if depth > 2:
            return self.heuristic_game_value(state)
        else:
            b = float('Inf')

            # Traverse through every possible successor
            for s in self.succ(state, self.opp):
                # Check if game_value is not 0
                if self.game_value(s) != 0:
                    return self.game_value(s), s
                

                val, curr = self.Max_Value(s, depth + 1) # Get max value

                # Check if max value is less than b
                if val < b:
                    b = val
                    min_state = s

        return b, min_state

    # Helper methods to find different position in two states
    def state_diff(self, state, next):
        pos = []
        for i in range(5):
            for j in range(5):
                if state[i][j] != next[i][j]:
                    dif = [i, j]
                    pos.append(dif)
        return pos

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.
        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.
                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).
        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = self.check_drop(state)  # Detect drop phase

        if not drop_phase:
            move = []
            a, next = self.Max_Value(state, 0)
            pos = self.state_diff(state, next)

            # Find previous position
            if state[pos[0][0]][pos[0][1]] == ' ':
                (pre_row, pre_col) = (pos[1][0], pos[1][1])
                (row, col) = (pos[0][0], pos[0][1])
            else:
                (pre_row, pre_col) = (pos[0][0], pos[0][1])
                (row, col) = (pos[1][0], pos[1][1])

            move.insert(0, (row, col))
            move.insert(1, (pre_row, pre_col))
            return move

        move = []
        a, next = self.Max_Value(state, 0)
        pos = self.state_diff(state, next)
        (row, col) = (pos[0][0], pos[0][1])

        while not state[row][col] == ' ':
            (row, col) = (pos[0][0], pos[0][1])

        # Ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))

        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition
        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.
        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][
                        col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        for row in range(2):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row + 1][i + 1] == state[row + 2][i + 2] == state[row + 3][i + 3]:
                    return 1 if state[row][i] == self.my_piece else -1

        # check / diagonal wins
        for row in range(3, 5):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row - 1][i + 1] == state[row - 2][i + 2] == state[row - 3][i + 3]:
                    return 1 if state[row][i] == self.my_piece else -1

        # check 2x2 box wins
        for row in range(4):
            for i in range(4):
                if state[row][i] != ' ' and state[row][i] == state[row][i + 1] == state[row + 1][i] == state[row + 1][i + 1]:
                    return 1 if state[row][i] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
