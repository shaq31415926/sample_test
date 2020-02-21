# numpy and pandas for data manipulation
import pandas as pd
import numpy as np
# visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# utilities
from itertools import chain
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class FeatureSelector(object):
    """
    Class for performing feature selection for machine learning or data preprocessing.

    Implements four different methods to identify features for removal

        1. Identifies features with a missing percentage greater than a specified threshold
        2. Identifies features with a single unique value
        2. Identifies features with a single unique value
        3. Identifies collinear variables with a correlation greater than a specified
        correlation coefficient
        4. Identifies features with the weakest relationship
        with the target variable
    """

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.missing_stats = None
        self.corr_matrix = None
        self.unique_stats = None
        self.record_missing = None
        self.record_collinear = None
        self.record_single_unique = None
        self.features = self.data.drop([self.target], axis=1)
        self.base_features = list(self.features)
        self.removed_features = None
        # Dictionary to hold removal operations
        self.ops = {}

    def identify_missing(self, missing_threshold):
        """Find the features with a fraction of missing values above `missing_threshold`"""
        self.missing_threshold = missing_threshold
        # Calculate the fraction of missing in each column
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(
            missing_series).rename(
            columns={'index': 'feature',
                     0: 'missing_fraction'})
        # Sort with highest number of missing values on top
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending=False)
        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(
            missing_series[missing_series > missing_threshold]).reset_index().rename(
            columns={'index': 'feature',
                     0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])
        self.record_missing = record_missing
        self.ops['missing'] = to_drop

        print('%d features with greater than %0.2f missing values.\n' % (
        len(self.ops['missing']), self.missing_threshold))

    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """
        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(
            unique_counts).rename(
            columns={'index': 'feature',
                     0: 'nunique'})

        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)
        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(
            unique_counts[unique_counts == 1]).reset_index().rename(
            columns={
                'index': 'feature',
                0: 'nunique'})

        to_drop = list(record_single_unique['feature'])
        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))

    def identify_collinear(self, correlation_threshold):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.
        """
        self.correlation_threshold = correlation_threshold
        self.corr_matrix = self.data.corr()
        high_corr_var = np.where(self.corr_matrix.abs() > correlation_threshold)
        high_corr_var = [(self.corr_matrix.index[x], self.corr_matrix.columns[y], round(self.corr_matrix.iloc[x, y], 2))
                         for x, y in zip(*high_corr_var) if x != y and x < y]

        if high_corr_var == []:
            record_collinear = pd.DataFrame()
            to_drop = []
        else:
            record_collinear = pd.DataFrame(
                high_corr_var).rename(
                columns={0: 'corr_feature',
                         1: 'drop_feature',
                         2: 'corr_values'})

            record_collinear = record_collinear.sort_values(by='corr_values', ascending=False)
            record_collinear = record_collinear.reset_index(drop=True)
            to_drop = list(record_collinear['drop_feature'])

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
        len(self.ops['collinear']), self.correlation_threshold))

    def identify_weakest_features(self, data, target, target_grouped, number_features):
        """
        statistical test to remove features with the weakest relationship
        with the target variable
        """
        X = data.drop([target, 'url_id', 'avg_ranking_score', target_grouped], axis=1)
        y = data[target]
        # apply SelectKBest class to extract top 20 best features
        bestfeatures = SelectKBest(score_func=chi2, k=20)
        fit = bestfeatures.fit(X, y)
        df_scores = pd.DataFrame(fit.scores_)
        df_columns = pd.DataFrame(X.columns)
        feature_scores = pd.concat([df_columns, df_scores], axis=1)
        feature_scores.columns = ['Features', 'Score']  # naming the dataframe columns
        feature_scores = feature_scores.sort_values(['Score'], ascending=False)  # sorting the dataframe by Score
        # print(feature_scores.nlargest(20,'Score'))  #print the 20 best features
        to_drop = list(feature_scores.nsmallest(number_features, 'Score')['Features'])
        self.feature_scores = feature_scores
        self.ops['weakest_features'] = to_drop
        data.drop(to_drop, axis=1, inplace=True)

        print(to_drop, '\n')
        print('%d features with the lowest score with the target variable removed\n' % (
            len(self.ops['weakest_features'])))
        print("new shape of data:", data.shape)

        return data

    def plot_unique(self):
        """Histogram of number of unique values in each feature"""
        if self.record_single_unique is None:
            raise NotImplementedError(
                'Unique values have not been calculated. Run `identify_single_unique`')

        # Histogram of number of unique values
        self.unique_stats.plot.hist(edgecolor='k', figsize=(7, 5))
        plt.ylabel('Frequency', size=14)
        plt.xlabel('Unique Values', size=14)
        plt.title('Number of Unique Values Histogram', size=16)

    def plot_missing(self):
        """Histogram of missing fraction in each feature"""
        if self.record_missing is None:
            raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")

        # Histogram of missing values
        plt.style.use('seaborn-white')
        plt.figure(figsize=(7, 5))
        plt.hist(self.missing_stats['missing_fraction'], bins=np.linspace(0, 1, 11), edgecolor='k', color='red',
                 linewidth=1.5)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel('Missing Fraction', size=14)
        plt.ylabel('Count of Features', size=14)
        plt.title("Fraction of Missing Values Histogram", size=16)

    def plot_collinear(self):
        """
        Heatmap of the correlation values. If plot_all = True plots all the correlations otherwise
        plots only those features that have a correlation above the threshold
        """
        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')

        corr_matrix_plot = self.corr_matrix
        title = "All Correlations"
        f, ax = plt.subplots(figsize=(10, 8))
        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap="Blues")
        plt.title(title, size=14)

    def remove(self, methods):
        """Remove the features from the data according to the specified methods."""
        features_to_drop = []
        data = self.data

        if methods == 'all':
            print('{} methods have been run\n'.format(list(self.ops.keys())))
            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))
        else:
            # Iterate through the specified methods
            for method in methods:
                # Check to make sure the method has been run
                if method not in self.ops.keys():
                    raise NotImplementedError('%s method has not been run' % method)
                # Append the features identified for removal
                else:
                    features_to_drop.append(self.ops[method])
            # Find the unique features to drop
            features_to_drop = set(list(chain(*features_to_drop)))

        features_to_drop = list(features_to_drop)
        # Remove the features and return the data
        data = data.drop(columns=features_to_drop)
        self.removed_features = features_to_drop
        print('Removed %d features.' % len(features_to_drop))
        print('\n', features_to_drop)

        return data

    def identify_all(self, selection_params):
        """
        Use all three of the methods to identify features to remove.
        """
        # Check for all required parameters
        for param in ['missing_threshold', 'correlation_threshold']:
            if param not in selection_params.keys():
                raise ValueError('%s is a required parameter for this method.' % param)

        # Implement each of the three methods
        self.identify_missing(selection_params['missing_threshold'])
        self.identify_single_unique()
        self.identify_collinear(selection_params['correlation_threshold'])
        # self.identify_weakest_features(selection_params['target'], selection_params['number_features'])

        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)

        print('%d total features out of %d identified for removal.\n' % (self.n_identified, self.data.shape[1]))

    def check_removal(self):
        """Check the identified features before removal. Returns a list of the unique features identified."""
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        print('Total of %d features identified for removal' % len(self.all_identified))

        return list(self.all_identified)
