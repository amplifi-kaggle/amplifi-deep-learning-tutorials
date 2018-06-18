import collections

def sparseVectorDotProduct(v1, v2):
    # BEGIN_YOUR_CODE
    return 0
    # END_YOUR_CODE
    
v1 = collections.defaultdict(float, {'a': 5})
v2 = collections.defaultdict(float, {'b': 2, 'a': 3})
print 1, sparseVectorDotProduct(v1, v2)

v1 = collections.defaultdict(float, {'c': 5})
v2 = collections.defaultdict(float, {'b': 1, 'a': 2})
print 2, sparseVectorDotProduct(v1, v2)

v1 = collections.defaultdict(float, {'a': 5, 'b': 4})
v2 = collections.defaultdict(float, {'b': 2, 'a': -1})
print 3, sparseVectorDotProduct(v1, v2)
