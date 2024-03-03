import numpy as np
import matplotlib.pyplot as plt
import math

def perception(x_list, y_list, T):
    """Dooes the perception algorithm on 2numpy matrixes and return the final weights
    after all the updates. Works"""
    #inicial condition for weights if the boundry isnt throught the origin
    theta_0 = 0
    both = x_list[0]*y_list[0]
    #init weights
    theta = np.zeros(both.shape)
    #start off with no errors
    number_of_errors = 0

    theta = theta.transpose()


    progression = []

    for t in range(T): # for T epochs as hyperparameter
        for xi, yi in zip(x_list, y_list):

            #print(xi.shape, yi.shape)

            #calculate the classifier which will return positive or negative label
            #classifier = yi * get_sign(xi @ theta) #+ theta_0
            classifier = (xi @ theta) #+ theta_0

            if  (classifier == 0).all() or (theta == 0).all():
                #preform update on weigths
                thetaprior = theta
                theta = yi*xi
                theta = theta + thetaprior
                print(f"in {t}, we progress to {theta}")
                progression.append(theta)

                #preform update on inicial condition for weights
                theta_0 = theta_0 + yi



                number_of_errors = number_of_errors + 1

    progression = [arr.tolist() for arr in progression]



    return number_of_errors, theta, progression, theta_0


def get_sign(x):
    if (x >= 0).all():return 1
    else: return -1




'''Notes and takeaways:

By starting at x2, we only make 1 mistake compared to NOE, not in general
in general, we make less mistakes




'''

def create_feature_vectors(d):
    """
    Creates a set of n=d labeled d-dimensional feature vectors with the specified values.
    """
    features = np.zeros((d, d))
    for t in range(d):
        for i in range(d):
            if i == t:
                features[t][i] = math.cos(2*math.pi*t)
            else:
                features[t][i] = 0
    return features







x_list = create_feature_vectors(2)

print(x_list)

y_list = np.array([[1],
                   [1]])

NOE, weights, progression, theta_0 = perception(x_list, y_list,10)




print( f"Number of errors{NOE}, weights {weights}, theta_0 {theta_0}")

'''for i in range(2):
    print( x_list[i] * weights[i])
'''
