import game
import random
import math
import mynumpy as mp

#==========================================================
# MinimaxAgent class
#==========================================================

class MinimaxAgent:
    def V(self, state):
        print '-',
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
            
        # raise Exception('Not implemented yet')
        # END_YOUR_CODE
        
    def policy(self, state):

        # BEGIN_YOUR_CODE

        #print [(self.V(game.get_next_state(state, action)), action) for action in game.get_possible_actions(state)]

        actions = game.get_possible_actions(state)

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            return actions[mp.argmax([self.V(game.get_next_state(state, action)) for action in actions])]
        else:
            return actions[mp.argmin([self.V(game.get_next_state(state, action)) for action in actions])]
        
        # raise Exception('Not implemented yet')
        # END_YOUR_CODE

#==========================================================
# Expectimax class
#==========================================================

class ExpectimaxAgent:
    def V(self, state):
        # BEGIN_YOUR_CODE
        if game.is_end(state):
            return game.utility(state)

        actions = game.get_possible_actions(state)

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            value = -game.INT_INF
            for action in actions:
                value = max(value, self.V(game.get_next_state(state, action)))
        else:
            value = 0.0
            for action in actions:
                value += (1/float(len(game.get_possible_actions(state)))) * self.V(game.get_next_state(state, action))

        return value
        # raise Exception('Not implemented yet')
        # END_YOUR_CODE
        
    def policy(self, state):
        # BEGIN_YOUR_CODE

        actions = game.get_possible_actions(state)

        player = game.get_player_from_state(state)
        if player == game.MAX_PLAYER:
            return actions[mp.argmax([self.V(game.get_next_state(state, action)) for action in actions])]
        else:
            import random
            return random.choice(actions)
        
        # raise Exception('Not implemented yet')
        # END_YOUR_CODE

#==========================================================
# Alpha-beta Pruning class
#==========================================================

class PruningMinimaxAgent:
    def V(self, state, alpha=-game.INT_INF, beta=game.INT_INF):
        print '-',
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
        # raise Exception('Not implemented yet')
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
            # return actions[mp.argmin([self.V(game.get_next_state(state, action)) for action in actions])]
        
        # raise Exception('Not implemented yet')
        # END_YOUR_CODE
