import util
import wordsegUtil

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def start_state(self):
        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def is_end(self, state):
        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def succ_and_cost(self, state):
        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
        
unigramCost, bigramCost = wordsegUtil.makeLanguageModels('leo-will.txt')
smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
possibleFills = wordsegUtil.makeInverseRemovalDictionary('leo-will.txt', 'aeiou')
problem = JointSegmentationInsertionProblem('mgnllthppl', smoothCost, possibleFills)

import backtracking_search
bts = backtracking_search.BacktrackingSearch(verbose=0)
# print bts.solve(problem)

import uniform_cost_search
ucs = uniform_cost_search.UniformCostSearch(verbose=0)
print ucs.solve(problem)