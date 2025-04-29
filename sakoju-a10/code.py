"""
Assignment 10, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""

import os
import sys
import random
from collections import Counter
import numpy as np
import time
import csv
from io import StringIO
from sklearn.decomposition import PCA, NMF
from abc import ABC, abstractmethod


def save_file(filename,actual, preds):
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Actual", "Predicted"])
        for a, pred in zip(actual, preds):
            csvwriter.writerow([a, pred])

def euclidean_distance(x1, x2):
    # return np.linalg.norm(x1-x2, ord=2)
    return np.sqrt(np.sum((x1-x2)**2))

class LearningAlgorithm:
    def __init__(self, name, *args, **kwargs):
        self.name = name
        
    def fit(self, X, y):
        return "Fit X_train and y_train"
        
    def fit_train(self, X, y):
        self.train()
        return "Fit X_train and y_train"

    def train(self):
        return []
    
    def predict(self, X_test):
        return None
    
    def predictx(self, x1):
        return None, None


class KNN(LearningAlgorithm):
    def __init__(self, name, k):
        super(KNN, self).__init__(name, k)
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.y_preds = []
    
    def predictx(self, x1):
        # print(x1, self.X_train[0])
        distances = [euclidean_distance(x1, x2) for x2 in self.X_train]
        k_nearest_idxs = np.argsort(distances)[:self.k]
        k_nearest_digits = [self.y_train[i] for i in k_nearest_idxs]
        mostcommon = Counter(k_nearest_digits).most_common()
        label = mostcommon[0][0]
        confidence = np.round(mostcommon[0][1]/self.k,3)
        # print(confidence, label)
        self.y_preds.append(label)
        return label, confidence
    
    def predict(self, X_test):
        predictions = [self.predictx(x) for x in X_test]
        return predictions

class PCAKNN(LearningAlgorithm):
    """Dimensionality reduction PCAs
    """
    def __init__(self, name, k):
        super(PCAKNN, self).__init__(name, k)
        self.k = k
        
    def fit(self, X, y):
        pca = PCA()
        pca.fit(X)
        self.X_train = pca.transform(X)
        self.y_train = y
        self.y_preds = []
    
    def predictx(self, x1):
        # print(x1, self.X_train[0])
        distances = [euclidean_distance(x1, x2) for x2 in self.X_train]
        k_nearest_idxs = np.argsort(distances)[:self.k]
        k_nearest_digits = [self.y_train[i] for i in k_nearest_idxs]
        mostcommon = Counter(k_nearest_digits).most_common()
        label = mostcommon[0][0]
        confidence = np.round(mostcommon[0][1]/self.k,3)
        # print(confidence, label)
        self.y_preds.append(label)
        return label, confidence
    
    def predict(self, X_test):
        pca = PCA()
        pca.fit(X_test)
        X_test_pca = pca.transform(X_test)
        predictions = [self.predictx(x) for x in X_test_pca]
        return predictions


class LMS(LearningAlgorithm):
        
    def __init__(self, name, alpha):
        super(LMS, self).__init__(name, alpha)
        self.w = None
        self.alpha = alpha
        
    def fit(self, X, y, y_test):
        self.X = X
        self.Y = y
        self.y_preds = []
        self.y_test = y_test
    
    # def check_matrix(self, mat, name):
    #     if len(mat.shape) != 2:
    #         raise ValueError(''.join(["Wrong matrix ", name]))
    
    # def add_basis(self, X):
    #     self.check_matrix(X, 'X')
    #     return np.hstack((np.ones((X.shape[0], 1)), X))
    
    # def train_LS(self, X, T):
    #     """Least Squares
    #     """
    #     #train w = (X.TX)^-1 X.T T  (derived from argmin sum of least squares)
    #     # dim of w is size of training samples (48K)
    #     X = self.add_basis(X)
    #     self.w = np.linalg.inv(X.T @ X) @ X.T @ T
    #     return self.w

    # def predict1(self, X):
    #     #for least square
    #     X = self.add_basis(X)
    #     Y = X @ self.w
    #     return Y

    def train(self, X, y):
        self.w = np.random.rand(X.shape[1])
        # iterations
        for i in range(1000):
            preds = np.dot(X, self.w)
            error = y - preds
            # update on all
            self.w += self.alpha * np.dot(X.T, error)
    
    def predict(self, X):
        preds = np.dot(X, self.w).astype(int)
        self.ypreds = preds.flatten()
        return preds
            
    # single layer ADALINE
    # def train_regular(self, X, T):
    #     """Least Mean Square method LMS
    #     """
    #     for i in range(X.shape[0]):
    #         self.train_step(X[i], T[i])
                
    # def train1(self, X, y):
    #     """Least Mean Square method LMS with Gradient descent rule
    #     """
    #     self.eta = 0.01
    #     self.epochs = 50
    #     self.cost = []
    #     self.w_ = np.zeros(1+X.shape[1])
    #     for t in range(self.epochs):
    #         for i in range(X.shape[0]):
    #             output = np.dot(X, self.w_[1:]) + self.w_[0] #net input
    #             errors = (y - output)
    #             self.w_[1:] += self.eta * X.T.dot(errors)
    #             self.w_[0] += self.eta * errors.sum()
    #             cost = (errors **2).sum() / 2.0
    #             self.cost.append(cost)
    #     return self
                
    # def activation(self, X):
    #     return np.dot(X, self.w_[1:]) + self.w_[0]

    # def predict_proba(self, X):
    #     x = self.activation(X)
    #     # print(x)
    #     #softmax to get probabilities
    #     exp_x = np.exp(x - np.max(x))
    #     probs = exp_x / np.sum(exp_x, axis=0)
    #     idx = np.argmax(probs)
    #     ypreds = max(probs)
    #     # print(probs, ypreds)
    #     self.y_preds.append(ypreds)
    #     return ypreds, probs[idx]
    
    # def train_step(self, x, y):
    #     """online step update
    #     """
    #     # x is 1d this is stepwise update (train)
    #     x = np.hstack([1,x])
    #     x = x.reshape(-1,1)
    #     if self.w is None:
    #         self.w = np.zeros((x.shape[0], 1))
    #     # Widrow-Hoff rule, I guess it is correct implementation
    #     # should be biased by larger values of x_j since multiplied by x_ij
    #     #with LMS loss w_j = w_j_t-1 - alpha (y_i - W.T x_i ) * x_ij
    #     self.w -= self.alpha * (self.w.T @ x - y) * x
    
    # def predict_regular(self, X):
    #     X = self.add_basis(X)
    #     y = X @ self.w
    #     y = y.flatten().astype(int)
    #     self.y_preds = y
    #     # print(y)
    #     return self.y_preds
    


# class BaseModel(ABC):
#     @abstractmethod
#     def train(self, X, T):
#         pass
    
#     @abstractmethod
#     def use(self, X):
#         pass
    
# class LinearModel(BaseModel):
    
#     def __init__(self):
#         super().__init__()
#         self.w = None
    
#     def check_matrix(self, mat, name):
#         if len(mat.shape) != 2:
#             raise ValueError(''.join(["Wrong matrix ", name]))
    
#     def add_basis(self, X):
#         self.check_matrix(X, 'X')
#         return np.hstack((np.ones((X.shape[0], 1)), X))


class NB(LearningAlgorithm):
    """Naive Bayes Classifier PI P(X_i|Y) with Naive assumption
        argmax PI P(x_i|y) P(y)
    """
    def __init__(self, name):
        super(NB, self).__init__(name)
        #2^d-1 entries for d samples for each label K (2^d-1), d binary features => exponential in d
        self.class_probs = {}
        self.feature_probs = {}
        self.y_preds = []
        
    def fit_train(self, X, y):
        nmf = NMF()
        X_nmf = nmf.fit_transform(X)
        H = nmf.components_
        X_new = np.dot(X_nmf, H)
        self.features = X
        self.labels = y
        self.train()
    
    def train(self):
        n = len(self.features)
        unique_class_labels = np.unique(self.labels)
        
        #calculate feature probs and class probabilities
        
        for label in unique_class_labels:
            self.class_probs[label] = np.sum(self.labels == label)/ n
            label_data = self.features[self.labels == label]
            self.feature_probs[label] = {}
            for idx in range(self.features.shape[1]):
                self.feature_probs[label][idx] = {}
                for intensity in np.unique(self.features):
                    self.feature_probs[label][idx][intensity] = {}
                    self.feature_probs[label][idx][intensity] = np.sum(label_data[:,idx] == intensity)/len(self.labels)

    def predictx(self, x1):
        """select maximum a posteriori 

        Args:
            x1 numpy array: 1D array of intentisities for this_sample 1D

        Returns:
            tuple: label_at_max_score, max_score
        """
        max_prob = -1
        selected_label = None
        for label, class_prob in self.class_probs.items():
            p = class_prob
            for idx, intensity in enumerate(x1):
                p *= self.feature_probs[label][idx][intensity]
            if p > max_prob:
                max_prob = p
                selected_label = label 
        self.y_preds.append(selected_label)
        return selected_label, max_prob
    
    def predict(self, X_test):
        nmf = NMF(n_components=32, init = "random", tol=5e3)
        X_text_nmf = nmf.fit_transform(X_test)
        H = nmf.components_
        X_test_new = np.dot(X_text_nmf, H)
        predictions = [self.predictx(x) for x in X_test]
        return predictions
    
    

class NBC(LearningAlgorithm):
    """Naive Bayes Classifier PI P(X_i|Y) with Naive assumption
        argmax PI P(x_i|y) P(y) MAP
    """
    def __init__(self, name):
        super(NBC, self).__init__(name)
        # 2 for binary and l for muticlass labels
        #2^d-1 entries for d samples for each label K (2^d-1), d binary features => exponential in d
        self.class_probs = {}
        self.feature_probs = {}
        self.y_preds = []
        self.num_classes = None
        
    def fit_train(self, X, y):
        pca = PCA()
        pca.fit(X)
        X_pca = pca.transform(X)
        self.features = X_pca
        self.labels = y
        self.num_classes = len(np.unique(y))
        self.unique_labels  =  np.unique(y)
        # np.zeros(self.num_classes)
        #{l:0 for l in np.unique(self.labels)}
        self.train()
    
    def train(self):
        n = len(self.features)
        #calculate feature probs and class probabilities
        
        for label in self.unique_labels:
            self.class_probs[label] = np.sum(self.labels == label) / n
            feat_label = self.features[self.labels == label]
            self.feature_probs[label] = (np.sum(feat_label, axis=0)+1) / (len(feat_label)+2)
        

    def predictx(self, x1):
        """predict
        Args:
            x1 numpy array: 1D array of intentisities for this_sample 1D
        Returns:
            tuple: label_at_max_score, max_score
        """
        max_prob = -1
        pred_label = None
        posterior_probs = {}
        for label in self.class_probs:
            prior_prob = self.class_probs[label]
            # print(self.feature_probs[label].shape, x1.shape)
            likelihood = np.prod(self.feature_probs[label]**x1 * \
                                    (1 - self.feature_probs[label]) ** (1-x1))
            posterior_probs[label] = prior_prob * likelihood
            
            pred_label = np.argmax(posterior_probs)
            max_prob = posterior_probs[pred_label] /np.sum(posterior_probs) \
                                        if np.sum(posterior_probs) > 0 else 1
        self.y_preds.append(pred_label)
        return pred_label, max_prob
    
    def predict(self, X_test):
        pca = PCA()
        pca.fit(X_test)
        X_test_pca = pca.transform(X_test)
        predictions = [self.predictx(x) for x in X_test_pca]
        return predictions
    
class MAP(LearningAlgorithm):
    """ Maximum a posteriori argmax P(theta|D)
        
    Args:
        LearningAlgorithm (MAP): MAP
    """
    def __init__(self, name, num_classes, img_size):
        super.__init__(name, num_classes, img_size)
        self.num_classes = num_classes
        self.img_size = img_size
        self.class_probs = np.zeros(num_classes)
        self.feature_probs = np.zeros((num_classes, img_size, img_size))
        self.y_preds = []
        
    def fit_train(self, X, y):
        self.images = X
        self.labels = y
        self.train()
    
    def train(self):
        #calculate feature probs and class probabilities
        
        for label in range(self.num_classes):
            self.class_probs[label] = np.sum(self.labels == label) / len(self.labels)
            this_class_features = self.images[self.labels == label]
            self.feature_probs[label] = \
                        np.sum(this_class_features, axis=0)/len(this_class_features)
    
    def predictx(self, image):
        """select maximum a posteriori 
            argmax p(y|x)
        Args:
            x1 numpy array: 1D array intentisities for this_image
        Returns:
            tuple: label_at_max_posteriori, max_score
        """
        posterior_probabilities = []
        for label in range(self.num_classes):
            prior_probability = np.log(self.class_probs[label])
            likelihood_prob = np.sum(image * np.log(self.feature_probs[label] + 1e-9) + \
                    (1-image) * np.log(1-self.feature_probs[label] + 1e-9) )
            posterios_prob = prior_probability + likelihood_prob
            posterior_probabilities.append(posterios_prob)
            
        pred_label = np.argmax(posterior_probabilities)
        confidence_score = np.exp(posterior_probabilities[pred_label] - np.max(posterior_probabilities))
        self.y_preds.append(pred_label)
        return pred_label, confidence_score
    
    def predict(self, X_test):
        predictions = [[self.predictx(x)] for x in X_test]
        return predictions

class MLE(LearningAlgorithm):
    """Maximum Likelihood P(D|theta)
    """
    def __init__(self, name, num_classes, img_size):
        super.__init__()
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.y_preds = []
        
    def fit_train(self, X, y):
        self.data = X
        self.labels = y
        self.unique_labels = np.unique(y)
        self.n = len(X)
        self.train()
    
    def train(self):
        #calculate feature probs and class probabilities
        
        for label in self.unique_labels:
            self.class_priors[label] = np.sum(self.labels == label) / self.n

        for label in self.unique_labels:
            label_featues = self.data[self.labels == label]
            self.feature_likelihoods[label] = {}
            for idx in range(self.data.shape[1]):
                self.feature_likelihoods[label][idx] = {}
                for intensity in np.unique(self.data[:, idx]):
                    count = np.sum(label_featues[:, idx] == intensity)
                    self.feature_likelihoods[label][idx][intensity] = (count+1) / (len(label_featues)+2)
    
    def predictx(self, image):
        """select maximum a posteriori 
            argmax p(y|x)
        Args:
            x1 numpy array: 1D array intentisities for this_image
        Returns:
            tuple: label_at_max_posteriori, max_score
        """
        posterior_probabilities = {}
        for label, prior in self.class_priors.items():
            likelihood = 1
            for idx, intensity in enumerate(image):
                likelihood *= self.feature_likelihoods[label][idx].get(intensity, 1e-6)
            posterior_probabilities[label] = prior * likelihood
            
        sum_post = sum(posterior_probabilities.values())
        norm_probs = {label: prob/ sum_post for label, prob in posterior_probabilities.items()}
        
        selected_label = max(norm_probs, key=norm_probs.get)
        score = norm_probs[selected_label]
        self.y_preds.append(selected_label)
        return selected_label, score
    
    def predict(self, X_test):
        predictions = [[self.predictx(x)] for x in X_test]
        return predictions
    
    def iter_train(self, X_test, init_labels, iterations=10):
        self.labels = init_labels
        for _ in range(iterations):
            predictions = self.predict(X_test)
            self.train(X_test, np.array(predictions))
        return self.labels
  

def main():
    args = sys.argv
    alg = "linear"
    # print(args)
    for i,arg in enumerate(args):
        if i == 0 or "code.py" in arg:
            continue
        elif i == 1 and ("knn" in arg or "linear" in arg or "nb" in args):
            alg = arg.strip()
        else:
            continue
        
    rawdata = sys.stdin.readlines()
    # print(all_lines)
    # print(alg)
    # print(rawdata)
    num_features, n, num_classes = [None, None, None]
    data = np.array([])
    if "attributes" in rawdata[0] and "values" in rawdata[0] and "classes" in rawdata[0]:
        header = rawdata[0]
        header = header.replace("attributes", "").replace("values", "").replace("classes", "")
        num_features, num_attribute_val, num_classes = [int(val.strip()) for val in header.split (",") ]
        num_features = int(num_features)
        num_attribute_val = int(num_attribute_val)
        num_classes = int(num_classes)
        n = 0
        if "training" in rawdata[1]:
            rawdata =  rawdata[2:]
        test_index = None 
        for i, line in enumerate(rawdata):
            if "test" in line:
                test_index = i+1
                break
        train = np.genfromtxt(rawdata[0:test_index-1], delimiter=" ")
        test = np.genfromtxt(rawdata[test_index:], delimiter=" ")
        # print(data, train, test)
        X_test =  test
        y_test =  np.asarray([0.0,1.0])
    else:
        n = len(rawdata)
        num_features = len(rawdata[0].split(" ")[0:-1])
        data = []
        # try:
        data = np.genfromtxt(rawdata, delimiter=" ")
        n_train = int(n*6/10)
        train = data[:n_train, ]
        test = data[n_train:,]
        # print(train.shape, test.shape)
        # X = train[:,:-1]
        # y = train[:,-1:]
        X_test =  test[:,1:]
        y_test =  test[:,0]
        
    X = train[:,1:]
    y = train[:, 0]

    if alg == "knn":
        knn = KNN("knn",34)
        knn.fit(X, y)
        predictions = knn.predict(X_test)
        save_file("knn"+str(int(time.time()*1000))+"csv",y_test, knn.y_preds)
        for val in predictions:
            # print(val)
            label, score = val
            print(f'{label} {score}')
    
    if alg == "pcaknn":
        knn = PCAKNN("knn",3)
        if X.shape[0] > 10000:
            knn.fit(X, y)
            predictions = knn.predict(X_test)
            save_file("knn"+str(int(time.time()*1000))+"csv",y_test, knn.y_preds)
            for val in predictions:
                # print(val)
                label, score = val
                print(f'{label} {score}')
        else:
            print("PCAKNN is for large number of samples only. Do not use that on tiny.data")
    
    elif alg == "linear":
        lms = LMS("lms",0.1)
        lms.fit(X, y, y_test)
        lms.train(X, y)
        predictions = lms.predict(X_test)
        save_file("lms"+str(int(time.time()*1000))+"csv",y_test, lms.y_preds)
        for label in predictions:
            print(f'{label} {0.0}')
            
    elif alg == "nb":
        nbc = NB("NaiveBayesClassifier")
        nbc.fit_train(X, y)
        predictions = nbc.predict(X_test)
        save_file("nb"+str(int(time.time()*1000))+"csv",y_test, nbc.y_preds)
        for label, score in predictions:
            print(f'{label} {score}')
    elif alg == "map":
        num_classes = np.unique(y)
        img_size = 200
        map1 = MAP("MAP", num_classes, img_size)
        map1.fit(X, y)
        predictions = map1.predict(X_test)
        save_file("map"+str(int(time.time()*1000))+"csv",y_test, map1.y_preds)
        for label, score in predictions:
            print(f'{label} {score}')

    elif alg == "mle":
        num_classes = np.unique(y)
        img_size = 200
        map1.fit(X, y)
        mle = MLE("MAP", num_classes, img_size)
        predictions = mle.predict(X_test)
        save_file("mle"+str(int(time.time()*1000))+"csv",y_test, mle.y_preds)
        for label, score in predictions:
            print(f'{label} {score}')
        
        
    
if __name__ == "__main__":
	main()
    