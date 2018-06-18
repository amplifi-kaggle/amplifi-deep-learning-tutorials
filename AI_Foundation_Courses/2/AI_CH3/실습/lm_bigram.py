import math, collections

SENTENCE_BEGIN = '-BEGIN-'

corpus = [
    'I am Sam',
    'Sam I am',
    'I do not like green'
]
    
# Counting
unigram_counts = collections.defaultdict(float)
bigram_counts = collections.defaultdict(float)

'''
for bi-gram,

    P(w|w') = count(w', w) / count(w')
    
'''

# BEGIN_YOUR_CODE

for sentence in corpus:
    words = [SENTENCE_BEGIN] + sentence.split()
    for word in words:
        unigram_counts[word] += 1

for sentence in corpus:
    words = [SENTENCE_BEGIN] + sentence.split()
    for word, next_word in zip(word[:-1], word[1:]):
        bigram_counts[(word, next_word)] += 1
    #for i in range(len(words)-1):
    #bigram_counts[(words[i], words[i+1])] += 1
        

# END_YOUR_CODE
        
# Bigram function
def bigram(prev_word, curr_word):
    # BEGIN_YOUR_CODE
    prob = bigram_counts[(prev_word, curr_word)] / unigram_counts[prev_word]
    assert prob > 0

    return -math.log(prob)
    # END_YOUR_CODE

# Printing results
print('\n- Bigram probabilities - ')
print('P(-BEGIN-, I) = %f'%bigram(SENTENCE_BEGIN, 'I'))    
print('P(-BEGIN-, Sam) = %f'%bigram(SENTENCE_BEGIN, 'Sam'))    
print('P(I, do) = %f'%bigram('I', 'do'))
print('P(like, green) = %f'%bigram('like', 'green'))
