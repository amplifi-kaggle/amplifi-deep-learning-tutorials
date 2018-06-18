import game
import random
import numpy as np
        
#==========================================================
# MinimaxAgent class
#==========================================================
        
class MinimaxAgent:
    def V(self, state):
        # BEGIN_YOUR_CODE
        raise Exception('Not implemented yet')
        # END_YOUR_CODE
        
    def policy(self, state):
        # BEGIN_YOUR_CODE
        raise Exception('Not implemented yet')
        # END_YOUR_CODE
            
#==========================================================
# Alpha-beta Pruning class
#==========================================================

class PruningMinimaxAgent:
    def V(self, state, alpha=-game.INT_INF, beta=game.INT_INF):
        # BEGIN_YOUR_CODE
        raise Exception('Not implemented yet')
        # END_YOUR_CODE
        
    def policy(self, state):
        # BEGIN_YOUR_CODE
        raise Exception('Not implemented yet')
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

class DepthLimitedMinimaxAgent:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        
    def V(self, state, depth):
        # BEGIN_YOUR_CODE
        raise Exception('Not implemented yet')
        # END_YOUR_CODE
        
    def policy(self, state):
        # BEGIN_YOUR_CODE
        raise Exception('Not implemented yet')
        # END_YOUR_CODE


            