import warnings

import numpy as np
import sklearn

import experiments
import learners


class KNNExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/KNN.py
        """original"""
        # params = {'KNN__metric': ['manhattan', 'euclidean', 'chebyshev'], 'KNN__n_neighbors': np.arange(1, 51, 3),
        #           'KNN__weights': ['uniform']}

        """new: changed metric and n_neighbors"""
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        "distance"
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
            KNN__metric is our distance functions: manhattan and euclidean
                manhattan: sum(|x - y|)
                euclidean: sqrt(sum((x - y)^2))
        "weights"
            uniform: uniform weights. All points in each neighborhood are weighted equally.
            distance: weight points by the inverse of their distance. 
                      in this case, closer neighbors of a query point will have a greater 
                      influence than neighbors which are further away.
        """
        params = {'KNN__metric': ['manhattan', 'euclidean'],
                  'KNN__n_neighbors': np.arange(1, 51, 5),
                  'KNN__weights': ['uniform']}

        complexity_param = {'name': 'KNN__n_neighbors',
                            'display_name': 'Neighbor count',
                            'values': np.arange(1, 51, 5)}

        best_params = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1:
        # best_params = {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'uniform'}
        #
        # Dataset 1:
        # best_params = {'metric': 'euclidean', 'n_neighbors': 4, 'weights': 'uniform'}

        learner = learners.KNNLearner(n_jobs=self._details.threads)
        if best_params is not None:
            learner.set_params(**best_params)

        """perform_experiment(ds, ds_name, ds_readable_name, clf, clf_name, clf_label, params, timing_params=None,
                       iteration_details=None, complexity_param=None, seed=0, threads=1,
                       iteration_lc_only=False, best_params=None, verbose=False)"""

        """pipe is built with
        pipe = Pipeline([('Scale', StandardScaler()),
                     ('KNN', learner)])
        """
        experiments.perform_experiment(self._details.ds,
                                       self._details.ds_name,
                                       self._details.ds_readable_name,
                                       learner,
                                       'KNN',
                                       'KNN',
                                       params,
                                       complexity_param=complexity_param,
                                       seed=self._details.seed,
                                       best_params=best_params,
                                       threads=self._details.threads,
                                       verbose=self._verbose)
