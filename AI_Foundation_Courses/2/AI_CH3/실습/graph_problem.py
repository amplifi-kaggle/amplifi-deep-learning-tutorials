'''
class GraphProblem():
    def __init__(self, states, distances):
        self.states = states
        self.distances = distances
        self.adj = {}
        
        for state in states:
            self.adj[state] = []
        for (u, v) in distances.keys():
            self.adj[u].append( (v, distances[(u, v)]) )

    def start_state(self):
        # BEGIN_YOUR_CODE
        return 'S'
        # END_YOUR_CODE

    def is_end(self, state):
        # BEGIN_YOUR_CODE
        return state == 'G'
        # END_YOUR_CODE

    def succ_and_cost(self, state):
        # BEGIN_YOUR_CODE
        results = []

        i = 0
        for (u, distance) in self.adj[state]:
            next_state = u
            action = i
            cost = distance
            results.append((action, next_state, cost))
            i += 1

        return results
        
        # END_YOUR_CODE
'''

class GraphProblem():
    def __init__(self, states, distances):
        self.states = states
        self.distances = distances

    def start_state(self):
        # BEGIN_YOUR_CODE
        return 'S'
        # END_YOUR_CODE

    def is_end(self, state):
        # BEGIN_YOUR_CODE
        return state == 'G'
        # END_YOUR_CODE

    def succ_and_cost(self, state):
        # BEGIN_YOUR_CODE
        results = []

        # for edge in distances:
        for next_state in states:
            if (state, next_state) in distances:
                action = state + '->' + next_state
                cost = distances[(state, next_state)]
                results.append((action, next_state, cost))

        return results
        
        # END_YOUR_CODE
        
states = ['S', 'A', 'B', 'C', 'D', 'E', 'G']
distances = {
    ('S', 'A'): 1,
    ('A', 'B'): 3,
    ('A', 'C'): 1,
    ('B', 'D'): 3,
    ('C', 'D'): 1,
    ('C', 'G'): 2,
    ('D', 'G'): 3,
    ('D', 'E'): 4,
    ('S', 'G'): 12,
}

problem = GraphProblem(states, distances)

import backtracking_search
bts = backtracking_search.BacktrackingSearch(verbose=3)
print bts.solve(problem)

import uniform_cost_search
ucs = uniform_cost_search.UniformCostSearch(verbose=3)
print ucs.solve(problem)
