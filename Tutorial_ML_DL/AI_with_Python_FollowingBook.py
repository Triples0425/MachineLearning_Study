#------------ chapter 1. ---------------------
#from sklearn import datasets

#house_prices = datasets.load_boston()

#print(house_prices.data)

#print(house_prices.target)
#------------ chapter 1. end ----------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cv
from sklearn.naive_bayes import GaussianNB


from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'

data = np.loadtxt(input_file, delimiter=' , ')
x, y = data[:, :-1], data[:, -1]

classifier = GaussianNB()

classifier.fit(x, y)

