from string import punctuation, digits
import numpy as np
import random
import math



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.

    EX:
    x = np.dot(theta.transpose(), feature_vector)
    loss = float(max(0, 1 - label*(x + theta_0)))
    """


    x = np.dot(feature_vector, theta) + theta_0
    loss = max(0, 1 - x * label)
    return loss



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """
    #Get
    y = np.dot(feature_matrix, theta) + theta_0
    losses = np.maximum(0.0, 1 - y * labels)
    return np.mean(losses)
    #
    # NOTE: the above gives the same result as the following line.  However, we
    # prefer to avoid doing linear algebra and other iteration in pure Python,
    # since Python lists and loops are slow.
    #
    # return np.mean([hinge_loss_single(feature_vector, label, theta, theta_0)
    #                 for (feature_vector, label) in zip(feature_matrix, labels)])
    #


'''
def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    h_classifier = np.dot(current_theta, feature_vector) + current_theta_0

    if (h_classifier*label).all() <= 0:
        theta = current_theta + feature_vector * label
        theta_0 = current_theta_0 + label
    else:
        theta = current_theta
        theta_0 = current_theta_0

    return (theta, theta_0)



'''




def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    h_classifier = np.dot(current_theta, feature_vector) + current_theta_0

    if h_classifier*label <= 0:
        theta = current_theta + feature_vector * label
        theta_0 = current_theta_0 + label
    else:
        theta = current_theta
        theta_0 = current_theta_0

    return (theta, theta_0)




def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix. hyperparameter like epochs

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """

    '''
    
    Form of paramters for a single step through perception algo
    
    feature_vector = np.array([1, 2, 3])
    label = 1
    current_theta = np.array([0, 0, 0])
    current_theta_0 = 0

    '''
    feature_matrix = np.array(feature_matrix)
    # Get order dim of the feature matrix to see how many feature vectors
    # The code will iterate through this in order, but in practice this is done stochastically
    nsamples = feature_matrix.shape[0]

    #Define inicial weights for the algo
    zero_shape = feature_matrix[1,:].shape
    theta = np.zeros(zero_shape)
    theta_0 = 0

    #for all hyperparameter length
    for t in range(T):
        for i in get_order(nsamples):

            feature_vector = feature_matrix[i, :] #Get feature vector from ith row in matrix
            theta, theta_0 = perceptron_single_step_update(feature_vector, labels[i], theta, theta_0)





    return (theta, theta_0)





def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: It is more difficult to keep a running average than to sum and
    divide.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    '''

        Form of paramters for a single step through perception algo

        feature_vector = np.array([1, 2, 3])
        label = 1
        current_theta = np.array([0, 0, 0])
        current_theta_0 = 0

        '''
    feature_matrix = np.array(feature_matrix)
    # Get order dim of the feature matrix to see how many feature vectors
    # The code will iterate through this in order, but in practice this is done stochastically
    nsamples = feature_matrix.shape[0]

    # Define inicial weights for the algo
    zero_shape = feature_matrix[1,:].shape
    theta = np.zeros(zero_shape)
    theta_0 = 0

    #inits
    summed_theta = 0
    summed_theta_0 = 0


    # for all hyperparameter length
    for t in range(T):
        for i in get_order(nsamples):
            feature_vector = feature_matrix[i, :]  # Get feature vector from ith row in matrix

            #Preform potential update
            theta, theta_0 = perceptron_single_step_update(feature_vector, labels[i], theta, theta_0)

            #Track sum, not tracking average
            summed_theta = summed_theta + theta
            summed_theta_0 = summed_theta_0 + theta_0

    #compute averages
    average_theta = summed_theta/(nsamples*T)
    average_theta_0 = summed_theta_0/(nsamples*T)

    return (average_theta, average_theta_0)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    #GIVEN: feature_vector, label, Lamba, eta,theta, theta_0

    classifier = label*(np.dot(theta, feature_vector) + theta_0)
    regularization = 1 - eta*L

    #c -1 = negtive when c<1
    if classifier <= 1:
        #Not technical :update the weights with a regulariation and learning rate clasification
        theta = theta*regularization + eta* label * feature_vector
        theta_0 = theta_0 + eta*label

    else: theta = regularization * theta

    #print (myAnswer)

    return (theta, theta_0)










def pegasos2(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.

    """

    #GIVEN: feature_matrix, labels, T, L
    #Find final theta and theta_0 after complete pegasos algo

    # Get order dim of the feature matrix to see how many feature vectors
    # The code will iterate through this in order, but in practice this is done stochastically

    feature_matrix = np.array(feature_matrix)

    nsamples = feature_matrix.shape[0]

    # Define inicial weights for the algo
    zero_shape = feature_matrix.shape[1]
    theta = np.zeros(zero_shape)
    theta_0 = 0




    updates_num = 0

    # for all hyperparameter length
    for t in range(T):
        for i in get_order(nsamples):
            #Get ith row feature vector
            feature_vector = feature_matrix[i]


            #Set eta in accordance with the number of updates
            eta = 1/(np.sqrt(updates_num+1))



            theta, theta_0 = pegasos_single_step_update(feature_vector, labels[i], L, eta, theta,theta_0)
            updates_num =updates_num + 1


    return theta, theta_0








def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lambda value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """

    # Get the number of feature vectors in the feature matrix
    feature_matrix = np.array(feature_matrix)

    nsamples = feature_matrix.shape[0]

    # Define initial weights for the algorithm
    zero_shape = feature_matrix[1,:].shape
    theta = np.zeros(zero_shape)
    theta_0 = 0


    # Initialize variable to keep track of the number of updates
    updates_num = 0

    # Iterate through the feature matrix T times
    for t in range(T):
        # Iterate through the feature vectors in a random order
        for i in get_order(nsamples):
            # Get the i-th row feature vector and its corresponding label
            feature_vector = feature_matrix[i]
            label = labels[i]

            # Set eta in accordance with the number of updates
            eta = 1/np.sqrt(updates_num + 1)

            # Update the weights using the Pegasos algorithm
            theta, theta_0 = pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)

            # Increment the number of updates
            updates_num += 1





    return theta, theta_0

'''
Answer with                 #eta = 1/math.sqrt(t)

[-0.09099022 -0.87175803 -0.25925022 -0.01001534 -0.20553331  0.04006049
 -0.77291064 -0.55911502 -0.05110979 -0.10661402] -1.0
 
 
 
Answer with math.sqrt(updates_num)
 [-0.06954744 -0.86162156 -0.2819555  -0.0081372  -0.21209376  0.01992519
 -0.79511222 -0.51131422 -0.10754056 -0.06050196] -0.7774785982356529
 
 
 [-0.06954744 -0.86162156 -0.2819555  -0.0081372  -0.21209376  0.01992519
 -0.79511222 -0.51131422 -0.10754056 -0.06050196] -0.7774785982356529
 
 
 correct output
  ['-0.0850387', '-0.7286435', '-0.3440130', '-0.0560494', '-0.0260993',
   '0.1446894', '-0.8172203', '-0.3200453', '-0.0729161', '0.1008662']

 
'''




#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse answer
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    labels = []


    feature_matrix = np.array(feature_matrix)
    for i in range(feature_matrix.shape[0]):
        feature_vector = feature_matrix[i, :]
        classifier = np.dot(theta, feature_vector) + theta_0



        if classifier >0: #positive classification
            labels.append(1)
            pass
        else: #negative classification
            labels.append(-1)
            pass
    labels = np.array(labels)
    return labels




def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, **kwargs):
    """
    Trains a linear classifier and computes accuracy. The classifier is
    trained on the train data. The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        classifier (callable): A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix (numpy.ndarray): A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix (numpy.ndarray): A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels (numpy.ndarray): A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels (numpy.ndarray): A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        kwargs: Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        A tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    # Train the classifier on the training data
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)

    # Predict the labels for the training and validation data
    train_predictions = np.sign(np.dot(train_feature_matrix, np.transpose(theta)) + theta_0)
    val_predictions = np.sign(np.dot(val_feature_matrix, np.transpose(theta)) + theta_0)

    # Compute the accuracy for the training and validation data
    train_accuracy = np.mean(train_predictions == train_labels)
    val_accuracy = np.mean(val_predictions == val_labels)

    # Return the accuracies as a tuple
    return train_accuracy, val_accuracy





def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=False):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    # Your code here

    with open("stopwords.txt", "r") as file:
        stopword = [line.strip() for line in file.readlines()]
    stopword = []
    indices_by_word = {}  # maps word to unique index
    for text in texts:

        #A list of lowercased words in the string, where punctuation and digits count as their own words.
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            #dictionart
            indices_by_word[word] = len(indices_by_word)
            #print("word", indices_by_word)
    #print("words", indices_by_word)

    return indices_by_word


import numpy as np


'''
def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word:
                continue
            if binarize:
                feature_matrix[i, indices_by_word[word]] = 1
            else:
                feature_matrix[i, indices_by_word[word]] = -1
    return feature_matrix'''


def extract_bow_feature_vectors(docs, vocabulary):
    feature_vectors = []
    for doc in docs:
        # count the number of occurrences of each word in the document
        word_counts = {}
        for word in doc:
            if word in vocabulary:
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
        # convert the dictionary of word counts to a feature vector
        feature_vector = [word_counts.get(word, 0) for word in vocabulary]
        feature_vectors.append(feature_vector)
    return feature_vectors




def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()




'''
feature_vector = np.array([[2,8],
                          [9,2],
                          [3,4]])
label = np.array([[1],
                  [-1],
                  [1]])

theta = np.zeros(feature_vector.transpose().shape)
theta_0 = 0

#Single hinge loss test
hinge_loss_single(feature_vector, label, theta, theta_0)'''
'''

feature_matrix = np.array( [[ 0.1837462,   0.29989789 ,-0.35889786, -0.30780561, -0.44230703, -0.03043835,
   0.21370063,  0.33344998, -0.40850817, -0.13105809],
 [ 0.08254096,  0.06012654 , 0.19821234,  0.40958367,  0.07155838, -0.49830717,
   0.09098162,  0.19062183, -0.27312663,  0.39060785],
 [-0.20112519, -0.00593087,  0.05738862,  0.16811148, -0.10466314, -0.21348009,
   0.45806193, -0.27659307,  0.2901038,  -0.29736505],
 [-0.14703536, -0.45573697, -0.47563745, -0.08546162 ,-0.08562345,  0.07636098,
  -0.42087389, -0.16322197, -0.02759763 , 0.0297091 ],
 [-0.18082261,  0.28644149, -0.47549449, -0.3049562,   0.13967768,  0.34904474,
   0.20627692,  0.28407868,  0.21849356, -0.01642202]])
labels = np.array([-1, -1, -1 , 1 ,-1])
T = 10
L =  0.1456692551041303

theta, bias = pegasos(feature_matrix, labels, T, L)

print(theta, bias)'''