from collections import Counter
from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import learning_curve
from pylab import rcParams
import seaborn as sns

class MLPipeline(object):
    """
    Class for performing feature selection based on a specific model.

    Outputs the following:
        1. Plots feature importance
        2. Identify low scoring/redundant features to drop
        3. Plot model scores
        4. Plot learning curves
    """

    def __init__(self, X_train,X_test, y_train, y_test, model):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

        self.feature_importance = None
        self.y_predict = None
        # Dictionary to hold removal operations
        self.ops = {}

        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = model.predict(self.X_test)
        self.y_pred_train = model.predict(self.X_train)

    def plot_confusion_matrix(self):
        """Plots the confusion matrix"""
        rcParams['figure.figsize'] = 5.85, 4
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)
        class_names = [0, 1]
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("bottom")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

    def plot_learning_curves(self):
        """Plots learning curves for model validation"""
        rcParams['figure.figsize'] = 5.85, 4
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.X_train,
            self.y_train,
            # Number of folds in cross-validation
            cv=5,
            # Evaluation metric
            scoring='accuracy',
            # Use all computer cores
            n_jobs=-1,
            shuffle=True,
            # 5 different sizes of the training set
            train_sizes=np.linspace(0.01, 1.0, 5))

        # Create means and standard deviations of training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        # Create plot
        plt.title("Learning Curves")
        plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()

        plt.show()

    def score(self):
        """some evaluation metrics"""
        score = {'Accuracy (Test Data)': round(self.model.score(self.X_test, self.y_test), 2),
                 'Accuracy (Train Data)': round(self.model.score(self.X_train, self.y_train), 2),
                 'F1-Score (Test Data)': round(f1_score(self.y_test, self.y_pred, average='weighted'), 2),
                 'F1-Score (Train Data)': round(f1_score(self.y_train, self.y_pred_train, average='weighted'), 2),
                 }

        return score

    def feature_importance_plot(self):
        """plots feature importance"""
        feat_importances = pd.Series(self.model.feature_importances_,
                                     index=self.X_train.columns)
        feat_importances.nlargest(20).plot(kind='barh')
        plt.show()


