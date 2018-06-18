import util

pq = util.PriorityQueue()

# BEGIN_YOUR_CODE

pq.update('A', 10)
pq.update('B', 20)
pq.update('C', 30)
pq.update('A', 5)

print pq.heap

# print first item
print pq.heap[0]

# remove first item
state, priority = pq.remove_min()
print state, priority

# END_YOUR_CODE
