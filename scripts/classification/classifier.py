from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SVMClassifier:
    def __init__(self, name= None, model_path= None, features_path=None, labels_path=None) -> None:
        self.name = name
        self.model = svm.SVC(kernel="linear", probability=True)
        self.model_path = model_path
        self.feature_vecs_path = features_path
        self.labels_path = labels_path
    
    def train(self, feature_vecs=None, labels=None):
        if feature_vecs is None:
            feature_vecs = np.loadtxt(self.feature_vecs_path)
            labels = np.loadtxt(self.labels_path)
        self.model.fit(feature_vecs, labels) 

    def predict(self, feature_vecs):
        probabilities = self.model.predict_proba(feature_vecs)
        prediction = self.model.predict(feature_vecs)
        return prediction, probabilities

    def load_model(self):
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

    def save_model(self):
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)
        

class RFClassifier:
    def __init__(self, name= None, model_path= None, features_path=None, labels_path=None) -> None:
        self.name = name
        self.model = RandomForestClassifier(n_estimators=20, max_features=None, max_samples=0.4)  # max features none -> use all features
        self.model_path = model_path
        self.feature_vecs_path = features_path
        self.labels_path = labels_path
    
    def train(self, feature_vecs=None, labels=None):
        if feature_vecs is None:
            feature_vecs = np.loadtxt(self.feature_vecs_path)
            labels = np.loadtxt(self.labels_path)
        self.model.fit(feature_vecs, labels) 

    def predict(self, feature_vecs):
        probabilities = self.model.predict_proba(feature_vecs)
        prediction = self.model.predict(feature_vecs)
        return prediction, probabilities

    def load_model(self):
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

    def save_model(self):
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)


if __name__ == "__main__":

    X = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/classifiers/rf_4_x.txt")))
    Y = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/classifiers/rf_4_y.txt")))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=Y, cmap='viridis')  # cmap can be changed to other color maps
    # fig.colorbar(scatter, ax=ax)
    # ax.set_xlabel('X0')
    # ax.set_ylabel('X1')
    # ax.set_zlabel('X2')
    # fig.show()
    # input()
    clf = RFClassifier(model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/classifiers/rf_4.pkl")))
    clf.load_model()
    # clf.train(X,Y)
    # clf.save_model()
    decision_tree = clf.model.estimators_[0]
    # Plot the tree
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree, feature_names=["hist_diff", "target_length", "target_width", "length_ratio", "width_ratio"],fontsize=10)
    plt.title('Decision Tree from Random Forest')
    plt.show()
    input("enter")

