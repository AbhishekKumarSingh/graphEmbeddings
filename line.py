import random
from collections import defaultdict
from scipy.io import loadmat
from itertools import izip
import numpy as np


def sigmoid(x):
    """
    Computes sigmoid function

    @param x : x is a matrix
    """
    x = 1/(1 + np.exp(-x))
    return x


def normalize_rows(x):
    """
    Row normalization function

    normalizes each row of a matrix to have unit lenght
    @param x : x is a matrix
    """
    row_sums = np.sqrt(np.sum(x**2,axis=1,keepdims=True))
    x = x / row_sums
    return x


def negSampObjFuncAndGrad(predicted, target, outputVectors, K=10):
    """
    Negative sampling cost function for word2vec models

    computes cost and gradients for one predicted word vector and one
    target word vector using the negative sampling technique.

    Inputs:
        - predicted: numpy ndarray, predicted word vector (\har{r} in
                     the written component)
        - target   : integer, the index of the target word
        - outputVectors: "output" vectors for all tokens
        - K: it is sample size

    Outputs:
        - cost: cross entropy cost for the softmax word prediction
        - gradPred: the gradient with respect to the predicted word vector
        - grad: the gradient with restpect to all the other word vectors
    """
    gradPred = np.zeros_like(predicted)
    grad = np.zeros_like(outputVectors)
    cost = 0
    z = sigmoid(outputVectors[target].dot(predicted))
    cost -= np.log(z)
    gradPred += outputVectors[target] * (z-1.)
    grad[target]+= predicted * (z-1.)

    for k in range(K):
        sampled_idx = random.randint(0, 4)
        z = sigmoid(outputVectors[sampled_idx].dot(predicted))
        cost -= np.log(1 - z)
        gradPred += z * outputVectors[sampled_idx]
        grad[sampled_idx] += z * predicted

    return cost, gradPred, grad


# def sgrad_desc(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
#     """
#     Stochastic Gradient Descent

#     Inputs:
#        - f: the function to optimize, should take a single argument
#             and yield two outputs, a cost and the gradient with
#             respect to the arguments
#        - x0: the initial point to start SGD from
#        - step: the step size for SGD
#        - iterations: total iterations to run SGD for
#        - postprocessing: postprocessing function for the parameters
#             if necessary. In the case of word2vec we will need to
#             normalize the word vectors to have unit length.
#        - PRINT_EVERY: specifies every how many iterations to output
#     Output:
#        - x: the parameter value after SGD finishes
#     """
#     # Anneal learning rate every several iterations
#     ANNEAL_EVERY = 20000

#     if useSaved:
#         start_iter, oldx, state = load_saved_params()
#         if start_iter > 0:
#             x0 = oldx;
#             step *= 0.5 ** (start_iter / ANNEAL_EVERY)

#         if state:
#             random.setstate(state)
#     else:
#         start_iter = 0

#     x = x0

#     if not postprocessing:
#         postprocessing = lambda x: x

#     expcost = None

#     for iter in xrange(start_iter + 1, iterations + 1):
#         x = postprocessing(x)
#         cost, grad = f(x)
#         x -= step * grad

#         if iter % PRINT_EVERY == 0:
#             print "iteration=%d cost=%f" % (iter, cost)

#         if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
#             save_params(iter, x)

#         if iter % ANNEAL_EVERY == 0:
#             step *= 0.5

#     return x


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.iteritems()}

matfile = "data/blogcatalog.mat"

# Load labels
mat = loadmat(matfile)
A = mat['network']
graph = sparse2graph(A)
labels_matrix = mat['group']

# lookup tabel for vertices
vertex = {}
index = 0
for node in graph:
    if node not in vertex:
        vertex[node] = index
        index += 1

# make edge matrix (dictionary)
edges = {}
for node in graph:
    for adj_node in graph[node]:
        v_pair = (vertex[node], vertex[adj_node])
        edges[v_pair] = 1

# print len(graph)


# Generally converges in linear time with number of edges
EDGE_COUNT = len(edges)
V = len(graph)
d = 20
bsample = 15
eta = 0.01
outputVectors = np.random.normal(0, 1, (V, d))
inputVectors = np.random.normal(0, 1, (V, d))

for step in xrange(0, EDGE_COUNT):
    # sample mini batch edges
    # and updates the model parameter.
    # for loop

    # sample b edges
    sample_edges = []
    for i in xrange(bsample):
        edge = random.choice(edges.keys())
        sample_edges.append(edge)

    tgradPred = np.zeros_like(inputVectors[0])
    tgradj = np.zeros_like(outputVectors[0])
    tgrad = np.zeros_like(outputVectors) # (K, d) dimension.

    for i, j in sample_edges:
        # normalize rows of inputs and output matrix
        inputVectors = normalize_rows(inputVectors)
        outputVectors = normalize_rows(outputVectors)

        cost, gradPred, grad = negSampObjFuncAndGrad(inputVectors[i], j, outputVectors)
        tgradPred += (gradPred * 1)#w[i][j]
        tgradj += (grad[j] * 1) #w[i][j]
        tgrad += (grad * 1)  #w[i][j]

    inputVectors[i] -= eta*tgradPred
    outputVectors[j] -= eta*tgradj
    outputVectors -= eta*tgrad

print "==========feature learning done========"
print "writing features to blogcatalog.embeddings...."
with open('data/blogcatalog.embeddings', 'wb') as f:
    embeddings = inputVectors + outputVectors
    import pickle
    pickle.dump(embeddings, f)

print "embeddings written to file"
# print inputVectors
# print outputVectors

import pprint

pkl_file = open('data/blogcatalog.embeddings', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

pkl_file.close()
