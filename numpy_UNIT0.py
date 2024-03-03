import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here

    A = np.random.rand(n,1)
    print(A)


    return A

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here

    A = np.random.rand(h, w)
    B = np.random.rand(h, w)
    s = A+B

    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """

    s = np.linalg.norm(A+B)
    return s

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    Z = np.tanh(np.matmul(weights.transpose(), inputs))
    return Z

def vector_function(x,y):
    def scalar_function(x, y):
        """
        Returns the f(x,y) defined in the problem statement.
        """
        # Your code here
        if x <= y:
            return x * y
        else:
            return x / y

    vecfoo = np.vectorize(scalar_function)

    return vecfoo(x,y)



#Debug excersise
def get_sum_metrics(predictions, metrics=[]):

    '''todo: The function get_sum_metrics takes two arguments:
        a prediction and a list of metrics to apply to the prediction
        (say, for instance, the accuracy or the precision).
        Note that each metric is a function, not a number. T
        he function should compute each of the metrics for the prediction and sum them.
        It should also add to this sum three default metrics
        , in this case, adding 0, 1 or 2 to the prediction.'''

    #Metric is a list of fucntions
    for i in range(3): #Add 3 functions to metric list

        #Add fi(x) = x + i where i == iterator
        metrics.append(lambda x: x + i)
        #Add a function to the metrics list

    sum_metrics = 0

    #Evaluate all functions in list of functions metrics
    for metric in metrics:
        #Evaluate function with predictions
        sum_metrics += metric(predictions)

    #return the sum of all the metrics returning their own predictions
    return sum_metrics

def get_sum_metrics2(prediction, metrics=[], default_metrics=None):
    if default_metrics is None:
        default_metrics = [lambda x: x, lambda x: x + 1, lambda x: x + 2]

    all_metrics = metrics + default_metrics
    sum_metrics = 0
    for metric in all_metrics:
        sum_metrics += metric(prediction)

    return sum_metrics




#neural_network()
'''
inputs = np.array([[2]
                ,  [1]])
weights = np.array([[2],
                    [3]])

output = neural_network(inputs,weights)

print(output, output.shape)'''




#vector_function()
'''
x = np.array([[2,1],
              [1, 9]])
y = np.array([[7,1],
              [6, 3]])
print(x, x.shape)

out = vector_function(x,y)

print(out, out.shape, "ot")
'''

#debug problem
'''
metrics = []
predictions = 20

response = get_sum_metrics(predictions, metrics)
response2 = get_sum_metrics(predictions, metrics)

print(response, response2)
'''



print(np.array([1,1,1,1,1]).shape)