import game
import random
import mynumpy as mp
        
#==========================================================
# MinimaxAgent class
#==========================================================
        
class MinimaxAgent:
    def V(self, state):
        # BEGIN_YOUR_CODE
        if game.is_end(state):
            return game.utility(state)
        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            value = -game.INT_INF
            for action in game.get_possible_actions(state):
                value = max(value, self.V(game.get_next_state(state, action)))
        else:
            value = game.INT_INF
            for action in game.get_possible_actions(state):
                value = min(value, self.V(game.get_next_state(state, action)))

        return value
        # END_YOUR_CODE
        
    def policy(self, state):
        # BEGIN_YOUR_CODE
        actions = game.get_possible_actions(state)

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            return actions[mp.argmax([self.V(game.get_next_state(state, action)) for action in actions])]
        else:
            return actions[mp.argmin([self.V(game.get_next_state(state, action)) for action in actions])]
        
        # END_YOUR_CODE
            
#==========================================================
# Alpha-beta Pruning class
#==========================================================

class PruningMinimaxAgent:
    def V(self, state, alpha=-game.INT_INF, beta=game.INT_INF):
        # BEGIN_YOUR_CODE
        if game.is_end(state):
            return game.utility(state)

        actions = game.get_possible_actions(state)

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            value = -game.INT_INF
            for action in actions:
                value = max(value, self.V(game.get_next_state(state, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha: break
        else:
            value = game.INT_INF
            for action in actions:
                value = min(value, self.V(game.get_next_state(state, action), alpha, beta))
                beta = min(beta, value)
                if beta <= alpha: break

        return value
        # END_YOUR_CODE
        
    def policy(self, state):
        # BEGIN_YOUR_CODE
        actions = game.get_possible_actions(state)

        alpha = -game.INT_INF
        beta  = game.INT_INF

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            values = []
            for action in actions:
                next_state = game.get_next_state(state, action)
                value = self.V(next_state, alpha, beta)
                values.append(value)
                alpha = max(alpha, value)
                if beta <= alpha: break
            idx = mp.argmax(values)
            return actions[idx]
            # return actions[mp.argmax([self.V(game.get_next_state(state, action)) for action in actions])]
        else:
            values = []
            for action in actions:
                next_state = game.get_next_state(state, action)
                value = self.V(next_state, alpha, beta)
                values.append(value)
                beta = min(beta, value)
                if beta <= alpha: break
            idx = mp.argmin(values)
            return actions[idx]
        # END_YOUR_CODE

#==========================================================
# DepthLimitedMinimaxAgent class
#==========================================================

heuristic_array = [
    [  0, -10, -100, -1000],
    [  10,  0,    0,     0],
    [ 100,  0,    0,     0],
    [1000,  0,    0,     0]
]

def eval(state):
    result = 0    
    for cond in game.WIN_CONDITIONS:
        maxs = mins = 0
        for loc in cond:
            if state[loc] == game.MAX_PLAYER:
                maxs += 1
            elif state[loc] == game.MIN_PLAYER:
                mins += 1
        result += heuristic_array[maxs][mins]
        
    return result

# minimax + depth limited (no alpha-beta pruning applied)
class DepthLimitedMinimaxAgent:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        
    def V(self, state, depth):
        # BEGIN_YOUR_CODE
        if game.is_end(state):
            return game.utility(state)
        if depth == 0:
            return eval(state)

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            value = -game.INT_INF
            for action in game.get_possible_actions(state):
                value = max(value, self.V(game.get_next_state(state, action), depth))
        else:
            value = game.INT_INF
            for action in game.get_possible_actions(state):
                value = min(value, self.V(game.get_next_state(state, action), depth - 1))

        return value
        # END_YOUR_CODE
        
    def policy(self, state):
        # BEGIN_YOUR_CODE
        actions = game.get_possible_actions(state)

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            values = []
            for action in actions:
                next_state = game.get_next_state(state, action)
                value = self.V(next_state, self.max_depth)
                values.append(value)
            idx = mp.argmax(values)
            return actions[idx]
        else:
            values = []
            for action in actions:
                next_state = game.get_next_state(state, action)
                value = self.V(next_state, self.max_depth)
                values.append(value)
            idx = mp.argmin(values)
            return actions[idx]
        
        # END_YOUR_CODE


            
