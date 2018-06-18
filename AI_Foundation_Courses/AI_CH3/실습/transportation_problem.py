
class TransportationProblem():
    def __init__(self, end_state):
        self.end_state = end_state

    def start_state(self):
        # BEGIN_YOUR_CODE
        return 1
        # END_YOUR_CODE

    def is_end(self, state):
        # BEGIN_YOUR_CODE
        return state == self.end_state
        # END_YOUR_CODE

    def succ_and_cost(self, state):
        # BEGIN_YOUR_CODE
        results = []

        # Walk action
        if state + 1 <= self.end_state:
            next_state = state + 1
            action = 'Walk'
            cost = 1
            results.append((action, next_state, cost))

        # Walk action
        if 2 * state <= self.end_state:
            next_state = 2 * state
            action = 'Tram'
            cost = 2
            results.append((action, next_state, cost))

        return results
        
        # END_YOUR_CODE

problem = TransportationProblem(7)

import backtracking_search
bts = backtracking_search.BacktrackingSearch(verbose=3)
# print bts.solve(problem)

import uniform_cost_search
ucs = uniform_cost_search.UniformCostSearch(verbose=3)
print ucs.solve(problem)
