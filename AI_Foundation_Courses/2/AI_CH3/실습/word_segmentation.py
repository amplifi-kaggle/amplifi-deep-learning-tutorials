import util
import wordsegUtil

'''
state: thisisnotmybeautifulhouse

action: t       state: hisisnotmybeautifulhouse
action: th      state: isisnotmybeautifulhouse
action: thi     state: sisnotmybeautifulhouse

My solution:

State: the remaining input sentence
Initial state: the input sentence
End state: empty string
Action: removing the substring from the front of the current state 
Cost: unigram (the removed string)

Alternative solution:

State: the number of characters used to construct words
Initial state: 0
End state: the length of the input sentence
Action: increasing the state value
Cost: unigram (the 

'''

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def start_state(self):
        # BEGIN_YOUR_CODE
        return self.query
        # END_YOUR_CODE

    def is_end(self, state):
        # BEGIN_YOUR_CODE
        return state == ''
        # END_YOUR_CODE

    def succ_and_cost(self, state):
        results = []
        # BEGIN_YOUR_CODE
        for i in range(len(state)):
            action      = state[:i+1]
            next_state  = state[i+1:]
            cost        = self.unigramCost(action)
            results.append((action, next_state, cost))

        return results
        # END_YOUR_CODE

unigramCost, bigramCost = wordsegUtil.makeLanguageModels('leo-will.txt')
problem = SegmentationProblem('thisisnotmybeautifulhouse', unigramCost)

import backtracking_search
bts = backtracking_search.BacktrackingSearch(verbose=0)
#print bts.solve(problem)

import uniform_cost_search
ucs = uniform_cost_search.UniformCostSearch(verbose=3)
print ucs.solve(problem)
