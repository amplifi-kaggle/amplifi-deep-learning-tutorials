import util
import wordsegUtil

'''

state = ('-BEGIN-', 'thts', 'm', 'n', 'th', 'crnr')

action = 'thats'

state: ('thats', 'm', 'n', 'th', 'crnr')

action: 'me' -> state: ('me', 'n', 'th', 'crnr')
action: 'me' -> state: ('me', 'n', 'th', 'crnr')
action: 'me' -> state: ('me', 'n', 'th', 'crnr')
action: 'me' -> state: ('me', 'n', 'th', 'crnr')


'''

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def start_state(self):
        # BEGIN_YOUR_CODE
        return (wordsegUtil.SENTENCE_BEGIN, ) + tuple(self.queryWords)
        # END_YOUR_CODE

    def is_end(self, state):
        # BEGIN_YOUR_CODE
        return len(state) == 1
        # END_YOUR_CODE

    def succ_and_cost(self, state):
        # BEGIN_YOUR_CODE
        results = []

        for action in self.possibleFills(state[1]):
            next_state = (action, ) + state[2:]
            cost = self.bigramCost(state[0], action)
            results.append((action, next_state, cost))

        # If there is no possible candidate, just leave it as it was.   
        if len(results) == 0:
            action = state[1]
            newState = (action, ) + state[2:]
            cost = self.bigramCost(state[0], action)
            results.append((action, next_state, cost))
            

        return results

        # END_YOUR_CODE
    
unigramCost, bigramCost = wordsegUtil.makeLanguageModels('leo-will.txt')
possibleFills = wordsegUtil.makeInverseRemovalDictionary('leo-will.txt', 'aeiou')
problem = VowelInsertionProblem('thts m n th crnr'.split(), bigramCost, possibleFills)

print problem.start_state()

import backtracking_search
bts = backtracking_search.BacktrackingSearch(verbose=0)
# print bts.solve(problem)

import uniform_cost_search
ucs = uniform_cost_search.UniformCostSearch(verbose=2)
print ucs.solve(problem)
