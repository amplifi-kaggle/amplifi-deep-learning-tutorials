import math, collections

corpus = [
    'I am Sam',
    'Sam I am',
    'I do not like green'
]

# Counting
unigram_counts = collections.defaultdict(float)
# BEGIN_YOUR_CODE

for sentence in corpus:
    words = sentence.split()
    for word in words:
        unigram_counts[word] += 1



# END_YOUR_CODE    

# Printing unigram counts
print('- Unigram counts -')
for word in unigram_counts:
    print('unigram_count[%s] = %d'%(word, unigram_counts[word]))

# Unigram function
# lower value is more frequent 
def unigram(word):
    # BEGIN_YOUR_CODE
    return -math.log(unigram_counts[word] / sum(unigram_counts.values()))
    # END_YOUR_CODE
        
# Printing results
print('\n- Unigram probabilities - ')
print('P(Sam) = %f'%unigram('Sam'))
print('P(I) = %f'%unigram('I'))
print('P(green) = %f'%unigram('green'))
