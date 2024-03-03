import numpy as np
import matplotlib.pyplot as plt

'''
You will begin by writing your loss function, a hinge-loss function. 
For this function you are given the parameters of your model  and . 

Additionally, you are given a feature matrix in which the rows are feature vectors and the columns are individual 
features, and a vector of labels representing the actual sentiment of the corresponding feature vector.

Hinge Loss on One Data Sample
1 point possible (graded)
First, implement the basic hinge loss calculation on a single data-point.
 Instead of the entire feature matrix, you are given one row, representing the feature vector of a single data sample,
  and its label of +1 or -1 representing the ground truth sentiment of the data sample.

Reminder: You can implement this function locally first, and run python test.py in your sentiment_analysis
 directory to validate basic functionality before checking against the online grader here.

'''