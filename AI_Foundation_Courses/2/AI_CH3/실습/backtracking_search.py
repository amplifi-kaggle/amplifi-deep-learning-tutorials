import util

class BacktrackingSearch(util.SearchAlgorithm):
    def __init__(self, verbose=0):
        self.verbose = verbose
        
    def recurrence(self, state, path, path_cost):
        if self.verbose >= 2:
            print 'state %s with path %s [%d]'%(state, path, path_cost)
        self.num_explored += 1

        # BEGIN_YOUR_CODE

        if self.problem.is_end(state):
            # update self.best_path, self.best_path_cost
            if self.best_path_cost is None or self.best_path_cost > path_cost:
                self.best_path = path
                self.best_path_cost = path_cost
            return 

        for (action, next_state, cost) in self.problem.succ_and_cost(state):
            extended_path = path + (action, )
            extended_path_cost = path_cost + cost
            self.recurrence(next_state, extended_path, extended_path_cost)

        # END_YOUR_CODE

    def solve(self, problem):
        # Not thread-safe
        self.problem = problem
        self.num_explored = 0
        self.best_path, self.best_path_cost = None, None
        self.explored = set()
        
        initial_state = problem.start_state()
        empty_path = ()
        self.recurrence(initial_state, empty_path, 0)
        
        return self.best_path, self.best_path_cost, self.num_explored
