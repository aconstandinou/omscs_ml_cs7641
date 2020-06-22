import numpy as np

import experiments
import learners


class ANNExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/ANN.py
        # Search for good alphas
        """
        alphas:
        - constrains/penalizes our max weights: high of 10 and diminishing drastically
        - recall: the larger the weight for an attr, the more it dominates, can lead to OVERFITTING
        """
        alphas = [10 ** -x for x in np.arange(-1, 9.01, 0.5)]

        # TODO: Allow for better tuning of hidden layers based on dataset provided
        d = self._details.ds.features.shape[1]
        """
        hiddens
        - based on the number of features (Xs, or attrs) in our data set
        - we test 1-2-3 layers using a multiple or division of # Xs.
        - ex: 23 attributes: test 11, 23, 46 hidden layer sizes
        """
        hiddens = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
        """
        https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
        learning_rates:
        - hyper-parameter that controls how much to change the model in response to the estimated error 
                each time the model weights are updated. 
        - a value too small may result in a long training process that could get stuck, 
        - a value too large may result in learning a sub-optimal set of weights too fast or an unstable training process
        - may be the most important hyper-parameter when configuring ANN.
        - small positive value, often in the range between 0.0 and 1.0.
        """
        learning_rates = sorted([(2**x)/1000 for x in range(8)]+[0.000001])

        """
        logistic: the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        relu: the rectified linear unit function, returns f(x) = max(0, x)
        """
        params = {'MLP__activation': ['relu', 'logistic'],
                  'MLP__alpha': alphas,
                  'MLP__learning_rate_init': learning_rates,
                  'MLP__hidden_layer_sizes': hiddens}

        timing_params = {'MLP__early_stopping': False}

        iteration_details = {'x_scale': 'log',
                             'params': {'MLP__max_iter': [2 ** x for x in range(12)] +
                                                         [2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]},
                             'pipe_params': timing_params}

        complexity_param = {'name': 'MLP__alpha',
                            'display_name': 'Alpha',
                            'x_scale': 'log',
                            'values': alphas}

        best_params = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1:
        # best_params = {'activation': 'relu', 'alpha': 1.0, 'hidden_layer_sizes': (36, 36),
        #                'learning_rate_init': 0.016}
        # Dataset 2:
        # best_params = {'activation': 'relu', 'alpha': 1e-05, 'hidden_layer_sizes': (16, 16),
        #                'learning_rate_init': 0.064}

        learner = learners.ANNLearner(max_iter=3000,
                                      early_stopping=True,
                                      random_state=self._details.seed,
                                      verbose=self._verbose)

        if best_params is not None:
            learner.set_params(**best_params)
        cv_best_params = experiments.perform_experiment(self._details.ds,
                                                        self._details.ds_name,
                                                        self._details.ds_readable_name,
                                                        learner,
                                                        'ANN',
                                                        'MLP',
                                                        params,
                                                        complexity_param=complexity_param,
                                                        seed=self._details.seed,
                                                        timing_params=timing_params,
                                                        iteration_details=iteration_details,
                                                        best_params=best_params,
                                                        threads=self._details.threads,
                                                        verbose=self._verbose)

        # TODO: This should turn OFF regularization
        of_params = cv_best_params.copy()
        of_params['MLP__alpha'] = 0
        if best_params is not None:
            learner.set_params(**best_params)
        learner = learners.ANNLearner(max_iter=3000,
                                      early_stopping=True,
                                      random_state=self._details.seed,
                                      verbose=self._verbose)

        experiments.perform_experiment(self._details.ds,
                                       self._details.ds_name,
                                       self._details.ds_readable_name,
                                       learner,
                                       'ANN_OF',
                                       'MLP',
                                       of_params,
                                       seed=self._details.seed,
                                       timing_params=timing_params,
                                       iteration_details=iteration_details,
                                       best_params=best_params,
                                       threads=self._details.threads,
                                       verbose=self._verbose,
                                       iteration_lc_only=True)
