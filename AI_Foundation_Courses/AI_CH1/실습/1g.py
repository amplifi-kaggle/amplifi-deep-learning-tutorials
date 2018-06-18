import collections

def incrementSparseVector(v1, scale, v2):
    # BEGIN_YOUR_CODE
    pass
    # END_YOUR_CODE

v1 = collections.defaultdict(float, {'a': 5})
scale = 2
v2 = collections.defaultdict(float, {'b': 2, 'a': 3})
incrementSparseVector(v1, 2, v2)
print 1, v1