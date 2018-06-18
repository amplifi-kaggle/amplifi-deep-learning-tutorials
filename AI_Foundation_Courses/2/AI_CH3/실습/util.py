import collections
import heapq

############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def start_state(self): raise NotImplementedError("Override me")

    # Return whether |state| is an end state or not.
    def is_end(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succ_and_cost(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    # First, call solve on the desired SearchProblem |problem|.
    # Then it should set two things:
    # - self.actions: list of actions that takes one from the start state to an end
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem): raise NotImplementedError("Override me")

# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.priorities = collections.defaultdict(int)  # Map from item to priority
        self.size = 0

    # Insert |item| into the heap with priority |newPriority|.
    # Return whether the priority queue was updated.
    def update(self, item, new_priority):
        if item not in self.priorities:
            heapq.heappush(self.heap, (new_priority, item))
            self.priorities[item] = new_priority
            return True
        else:
            if new_priority < self.priorities[item]:
                self.heap.remove((self.priorities[item], item))
                heapq.heappush(self.heap, (new_priority, item))
                self.priorities[item] = new_priority
                return True
            return False
            
        heapq.heappush(self.heap, (new_priority, item))
        return is_updated

    # Returns (item with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def remove_min(self):
        try:
            priority, item = heapq.heappop(self.heap)
            del self.priorities[item]
            return (item, priority)
        except IndexError:
            return (None, None)
        return (None, None)
        
    def is_empty(self):
        return len(self.heap) == 0