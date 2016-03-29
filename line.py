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
        sampled_idx = dataset.sampleTokenIdx() # need to understand this; random.randint(0, 4)
        z = sigmoid(outputVectors[sampled_idx].dot(predicted))
        cost -= np.log(1 - z)
        gradPred += z * outputVectors[sampled_idx]
        grad[sampled_idx] += z * predicted
    
    return cost, gradPred, grad


def sgrad_desc(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    """
    Stochastic Gradient Descent
    
    Inputs:                                                         
       - f: the function to optimize, should take a single argument
            and yield two outputs, a cost and the gradient with
            respect to the arguments                            
       - x0: the initial point to start SGD from                     
       - step: the step size for SGD                                 
       - iterations: total iterations to run SGD for                 
       - postprocessing: postprocessing function for the parameters  
            if necessary. In the case of word2vec we will need to    
            normalize the word vectors to have unit length.          
       - PRINT_EVERY: specifies every how many iterations to output  
    Output:                                                         
       - x: the parameter value after SGD finishes                   
    """
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
            
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x: x
    
    expcost = None
    
    for iter in xrange(start_iter + 1, iterations + 1):
        x = postprocessing(x)
        cost, grad = f(x)
        x -= step * grad
        
        if iter % PRINT_EVERY == 0:
            print "iteration=%d cost=%f" % (iter, cost)
        
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
            
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    
    return x