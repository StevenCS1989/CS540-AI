import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    distance = 0 # Variable for manhattan distance

    # Loop through from_state
    for i in range(len(from_state)):
        # Check if the tile at index i is not at the correct spot and that it's non-zero
        if (from_state[i] != to_state[i]) and (from_state[i] != 0):
            x1, y1 = i // 3, i % 3 # Get the x-y values of that number from from_state
            x2, y2 = to_state.index(from_state[i]) // 3, to_state.index(from_state[i]) % 3 # Get the x-y values from to_state of the number in from_state
            distance += abs(x1 - x2) + abs(y1 - y2) # Do the manhattan distance equation and add it to distance

    return distance

def print_succ(state):
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    succ_states = []

    for i in range(len(state)):
        # Check if the current tile is empty
        if state[i] == 0:
            curr_row = i // 3 # Get the row of the tile
            curr_col = i % 3 # Get the column of the tile

            # UP
            # Check if the upper adjacent tile is within the boundry and non-zero
            if (0 <= (curr_row - 1) < 3) and (state[i - 3] != 0):
                # Swap values
                succ_state = state.copy() # Make a new list with same list as state
                temp = succ_state[i - 3] # Set temp variable to the adjacent upper tile
                succ_state[i - 3] = succ_state[i] # Swap
                succ_state[i] = temp # Swap

                succ_states.append(succ_state) # Append to list
            
            # DOWN
            # Check if the lower adjacent tile is within the boundry and non-zero
            if (0 <= (curr_row + 1) < 3) and (state[i + 3] != 0):
                # Swap values
                succ_state = state.copy() # Make a new list with same list as state
                temp = succ_state[i + 3] # Set temp variable to the adjacent upper tile
                succ_state[i + 3] = succ_state[i] # Swap
                succ_state[i] = temp # Swap

                succ_states.append(succ_state) # Append to list

            # LEFT
            # Check if the left adjacent tile is within the boundry and non-zero
            if (0 <= (curr_col - 1) < 3) and (state[i - 1] != 0):
                # Swap values
                succ_state = state.copy() # Make a new list with same list as state
                temp = succ_state[i - 1] # Set temp variable to the adjacent upper tile
                succ_state[i - 1] = succ_state[i] # Swap
                succ_state[i] = temp # Swap

                succ_states.append(succ_state) # Append to list

            # RIGHT
            # Check if the right adjacent tile is within the boundry and non-zero
            if (0 <= (curr_col + 1) < 3) and (state[i + 1] != 0):
                # Swap values
                succ_state = state.copy() # Make a new list with same list as state
                temp = succ_state[i + 1] # Set temp variable to the adjacent upper tile
                succ_state[i + 1] = succ_state[i] # Swap
                succ_state[i] = temp # Swap

                succ_states.append(succ_state) # Append to list

    return sorted(succ_states)

def print_move(current_state, h, g):
    print(f'{current_state} h={h} moves: {g}')

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    # Initialize the open set and closed set
    open_set = []
    closed_set = set()

    # Initialize the parent dictionary to store the parent node of each visited state
    parent = {}

    # Calculate the heuristic value for the initial state
    h = get_manhattan_distance(state, goal_state)
    g = 0

    # Step 1
    heapq.heappush(open_set, (h, g, state)) # Initial state

    # Initialize variables to keep track of the maximum size of the open set and the number of moves
    max_queue_size = 0
    num_moves = 0

    # Loop until the open set is empty
    while open_set: # Step 2
        # Step 3
        h, g, current_state = heapq.heappop(open_set)
        closed_set.add(tuple(current_state)) # Add node to closed_set

        # Step 4
        if current_state == goal_state:
            path = [current_state]
            while tuple(current_state) in parent:
                current_state = parent[tuple(current_state)]
                path.append(current_state)
            for i, state in enumerate(reversed(path)):
                print(f"{state} h={get_manhattan_distance(state, goal_state)} moves: {i}")
            print(f"Max queue length: {max_queue_size}")
            return

        # Step 5
        succ_states = get_succ(current_state)
        for succ_state in succ_states:
            # Calculate g and h values for the successor state
            new_g = g + 1
            new_h = get_manhattan_distance(succ_state, goal_state)

            # Check if the successor state is already in closed_set
            succ_tuple = tuple(succ_state)
            if succ_tuple in closed_set:
                continue

            # Check if the successor state is already in open_set
            found = False
            if len(open_set) > 0:
                for open_h, open_g, open_state in open_set:
                    if succ_state == open_state and new_g < open_g:
                        open_set.remove((open_h, open_g, open_state))
                        heapq.heappush(open_set, (new_h, new_g, succ_state))
                        parent[succ_tuple] = current_state
                        found = True
                        break
            if found:
                continue
        
            # Add the successor state to open_set
            heapq.heappush(open_set, (new_h, new_g, succ_state))
            parent[succ_tuple] = current_state

        # Update max_queue_size and num_moves
        max_queue_size = max(max_queue_size, len(open_set))
        num_moves += 1



if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """

    # This tests the manhatten distance
    print(get_manhattan_distance([2,5,1,4,3,6,7,0,0], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    solve([3, 4, 6, 0, 0, 1, 7, 2, 5])

