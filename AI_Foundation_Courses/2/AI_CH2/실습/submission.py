#!/usr/bin/python

import random
import collections
import math
import sys
import os, operator
from collections import Counter

############################################################
# Util
############################################################

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def readExamples(path):
    '''
    Reads a set of training examples.
    '''
    examples = []
    for line in open(path):
        # Format of each line: <output label (+1 or -1)> <input sentence>
        y, x = line.split(' ', 1)
        examples.append((x.strip(), int(y)))
    print 'Read %d examples from %s' % (len(examples), path)
    return examples

def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

def outputWeights(weights, path):
    print "%d weights" % len(weights)
    out = open(path, 'w')
    for f, v in sorted(weights.items(), key=lambda(f, v): -v):
        print>>out, '\t'.join([f, str(v)])
    out.close()

def verbosePredict(phi, y, weights, out):
    yy = 1 if dotProduct(phi, weights) > 0 else -1
    if y:
        print>>out, 'Truth: %s, Prediction: %s [%s]' % (y, yy, 'CORRECT' if y == yy else 'WRONG')
    else:
        print>>out, 'Prediction:', yy
    for f, v in sorted(phi.items(), key=lambda(f, v): -v * weights.get(f, 0)):
        w = weights.get(f, 0)
        print>>out, "%-30s%s * %s = %s" % (f, v, w, v * w)
    return yy

def outputErrorAnalysis(examples, featureExtractor, weights, path):
    out = open(path, 'w')
    for x, y in examples:
        print>>out, '===', x
        verbosePredict(featureExtractor(x), y, weights, out)
    out.close()

def interactivePrompt(featureExtractor, weights):
    while True:
        print '> ',
        x = sys.stdin.readline()
        if not x: break
        phi = featureExtractor(x) 
        verbosePredict(phi, None, weights, sys.stdout)

############################################################

def generateClusteringExamples(numExamples, numWordsPerTopic, numFillerWords):
    '''
    Generate artificial examples inspired by sentiment for clustering.
    Each review has a hidden sentiment (positive or negative) and a topic (plot, acting, or music).
    The actual review consists of 2 sentiment words, 4 topic words and 2 filler words, for example:

        good:1 great:1 plot1:2 plot7:1 plot9:1 filler0:1 filler10:1

    numExamples: Number of examples to generate
    numWordsPerTopic: Number of words per topic (e.g., plot0, plot1, ...)
    numFillerWords: Number of words per filler (e.g., filler0, filler1, ...)
    '''
    sentiments = [['bad', 'awful', 'worst', 'terrible'], ['good', 'great', 'fantastic', 'excellent']]
    topics = ['plot', 'acting', 'music']
    def generateExample():
        x = Counter()
        # Choose 2 sentiment words according to some sentiment
        sentimentWords = random.choice(sentiments)
        x[random.choice(sentimentWords)] += 1
        x[random.choice(sentimentWords)] += 1
        # Choose 4 topic words from a fixed topic
        topic = random.choice(topics)
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        # Choose 2 filler words
        x['filler' + str(random.randint(0, numFillerWords-1))] += 1
        return x

    random.seed(42)
    examples = [generateExample() for _ in range(numExamples)]
    return examples

def outputClusters(path, examples, centers, assignments):
    '''
    Output the clusters to the given path.
    '''
    print 'Outputting clusters to %s' % path
    out = open(path, 'w')
    for j in range(len(centers)):
        print>>out, '====== Cluster %s' % j
        print>>out, '--- Centers:'
        for k, v in sorted(centers[j].items(), key=lambda(k, v): -v):
            if v != 0:
                print>>out, '%s\t%s' % (k, v)
        print>>out, '--- Assigned points:'
        for i, z in enumerate(assignments):
            if z == j:
                print>>out, ' '.join(examples[i].keys())
    out.close()


############################################################
# Sentiment Classification
############################################################

random.seed(42)

# Problem B: extractWordFeatures

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE 
    
    phi = {}
    for word in x.split():
        if word not in phi:
            phi[word] = 0
        phi[word] += 1
    
    return phi
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

""" 
print('-- Test b0 --')
output = extractWordFeatures("a b a")
answer = {"a":2, "b":1}
print "output : %s" % output
print "answer : %s" % answer

print('-- Test b1 --')
for i in range(10):
    sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
output = extractWordFeatures(sentence)
print "output : %s" % output
"""

# Problem C: learnPredictor

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}
    # BEGIN_YOUR_CODE
    
    def predictor(x):
        if dotProduct(featureExtractor(x), weights) > 0:
            return 1
        else:
            return -1
    
    # weight vector update
    for t in range(numIters):
        for trainSample in trainExamples:
            x = trainSample[0]
            y = trainSample[1]
            phi = featureExtractor(x)
            
            # HingeLoss 값을 계산
            HingeLoss = max(0, 1 - dotProduct(weights, phi) * y)
        
            if HingeLoss > 0:
                #increment(weights, eta * y, phi)
                for feature in phi.keys():
                    if feature not in weights:
                        weights[feature] = 0
                    weights[feature] += eta * phi[feature] * y
                    
            
        training_error = evaluatePredictor(trainExamples, predictor)
        test_error = evaluatePredictor(testExamples, predictor)
        print("%d-th iteration: train error = %.2f, test error = %.2f" % (t, training_error, test_error))
    
    
    # END_YOUR_CODE
    return weights


print('-- Test c0 --')
trainExamples = (("pretty good", 1), ("bad plot", -1), ("not good", -1), ("pretty scenery", 1))
testExamples = (("pretty", 1), ("bad", -1))
weights = learnPredictor(trainExamples, testExamples, extractWordFeatures, numIters=1, eta=1)
print(weights)


print('-- Test c1 --')
trainExamples = (("hello world", 1), ("goodnight moon", -1))
testExamples = (("hello", 1), ("moon", -1))
weights = learnPredictor(trainExamples, testExamples, extractWordFeatures, numIters=20, eta=0.01)
print("weight for `hello\' : %f (should > 0)" % weights['hello'])
print("weight for `moon\' : %f (should < 0)" % weights['moon'])

print('-- Test c2 --')
trainExamples = readExamples('polarity.train')
devExamples = readExamples('polarity.dev')
weights = learnPredictor(trainExamples, devExamples, extractWordFeatures, numIters=20, eta=0.01)
outputWeights(weights, 'weights')
outputErrorAnalysis(devExamples, extractWordFeatures, weights, 'error-analysis')  # Use this to debug
trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(extractWordFeatures(x), weights) >= 0 else -1))
testError = evaluatePredictor(devExamples, lambda x : (1 if dotProduct(extractWordFeatures(x), weights) >= 0 else -1))
print("Final train error = %s (should < 0.04)" % trainError)
print("Final test error = %s (should < 0.30)" % testError)

# Problem D: generateExample

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE
        
        # 1. select words to be appeared
        
        words = random.sample( weights.keys(), random.randint(1, len(weights.keys())))
        
        # 2. generate a feature vector based on # 1.
        # phi = { key : random.randint(1, 3) for key in words}
        phi = {}
        
        for word in words:
            phi[word] = random.randint(1, 3)
            
        # 3. calcalate the label based on the given weight vector

        if dotProduct(weights, phi) >= 0:
            y = 1
        else:
            y = -1
        
        
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]


print('-- Test d0 --')
weights = {"hello":1, "world":1}
data = generateDataset(5, weights)
for datapt in data:
    if (dotProduct(datapt[0], weights) >= 0) != (datapt[1] == 1):
        print "Wrong Implementation"

print('-- Test d1 --')
weights = {}
for i in range(100):
    weights[str(i + 0.1)] = 1
data = generateDataset(100, weights)
for datapt in data:
    if dotProduct(datapt[0], weights) == 0:
        print "Wrong Implementation"


# Problem F: extractCharacterFeatures

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE

        phi = {}

        text = ''.join(x.split())

        for i in range(len(text) - (n - 1)):
            if text[i : i + n] not in phi:
                phi[text[i : i + n]] = 0
            phi[text[i : i + n]] += 1
        return phi
        
        # END_YOUR_CODE
    return extract


print('-- Test f0 --')
fe = extractCharacterFeatures(3)
sentence = "hello world"
output = fe(sentence)
answer = {"hel":1, "ell":1, "llo":1, "low":1, "owo":1, "wor":1, "orl":1, "rld":1}
print "output : %s" % output
print "answer : %s" % answer


print('-- Test c2 --')
fe = extractCharacterFeatures(5)
trainExamples = readExamples('polarity.train')
devExamples = readExamples('polarity.dev')
weights = learnPredictor(trainExamples, devExamples, fe, numIters=20, eta=0.01)
outputWeights(weights, 'weights-fe5')
outputErrorAnalysis(devExamples, fe, weights, 'error-analysis-fe5')  # Use this to debug
trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(fe(x), weights) >= 0 else -1))
testError = evaluatePredictor(devExamples, lambda x : (1 if dotProduct(fe(x), weights) >= 0 else -1))
print("Final train error = %s (should < 0.04)" % trainError)
print("Final test error = %s (should < 0.30)" % testError)


############################################################
# k-means Clustering
############################################################

# Problem M: kmeans

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE

    def square_dist(x1, x2):
        dist = 0.0
        features = set(x1.keys() + x2.keys())
        for feature in features:
            dist += (x1.get(feature, 0) - x2.get(feature, 0)) ** 2
        return dist

    centroids = random.sample( examples, K )

    N = len(examples)
    assigns = [0 for i in range(N)]
    old_loss = 0

    for iter in range(maxIters):
        loss = 0

        # Assignment step

        # First possible implementation
        for i in range(N):
            sample = examples[i]

            assigns[i] = 0
            
            mindist = square_dist(sample, centroids[0])
            for j in range(K):
                tempdist = square_dist(sample, centroids[j])
                if mindist > tempdist:
                    mindist = tempdist
                    assigns[i] = j

            loss += mindist

        if loss == old_loss:
            break
        else:
            old_loss = loss

        # Second possible implementation
        # distlist = [square_dist(sample, centroids[k]) for k in range(K)]
        # assigns[i] = distlist.index(min(distlist))

        # Update step
        centroids = [{} for k in range(K)]
        nClusters = [0 for k in range(K)]
        
        for i in range(N):
            increment(centroids[assigns[i]], 1, examples[i])
            nClusters[assigns[i]] += 1

        for k in range(K):
            # TODO: centroids[k] / nClusters[k]
            for feature in centroids[k].keys():
                centroids[k][feature] /= float(nClusters[k])

                
    return centroids, assigns, loss
    
    # END_YOUR_CODE


print('-- Test m0 --')
x1 = {0:0, 1:0}
x2 = {0:0, 1:1}
x3 = {0:0, 1:2}
x4 = {0:0, 1:3}
x5 = {0:0, 1:4}
x6 = {0:0, 1:5}
examples = [x1, x2, x3, x4, x5, x6]
centroids, assignments, totalCost = kmeans(examples, 2, maxIters=10)
# (there are two stable centroid locations)
print centroids
print assignments
print "output square loss : %.1f" % totalCost
print "answer square loss : 4.0 or 5.5"

print('-- Test m1 --')
K = 6
bestCenters = None
bestAssignments = None
bestTotalCost = None
examples = generateClusteringExamples(numExamples=1000, numWordsPerTopic=3, numFillerWords=1000)
centers, assignments, totalCost = kmeans(examples, K, maxIters=100)
outputClusters('output-cluster', examples, centers, assignments)

