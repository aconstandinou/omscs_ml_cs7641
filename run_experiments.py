import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import datetime as dt
import os

from datetime import datetime
import logging
import experiments

import load_titanic
import load_credit_default_data as ld

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_details:
        exp = experiment(details, verbose=verbose)

        logger.info("Running {} experiment: {}".format(timing_key, details.ds_readable_name))
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds

#
# def run_experiments_old(train_x, train_y, test_x):
#     """
#     Run our experiments on all Learners for our data sets
#     :return:
#     """
#     # LOGISTIC REGRESSION
#     logreg = LogisticRegression()
#     logreg.fit(train_x, train_y)
#     # Y_pred = logreg.predict(X_test)
#     acc_log = round(logreg.score(train_x, train_y) * 100, 2)
#     print(acc_log)
#
#     # SVM - Support Vector Machines
#     svc = SVC()
#     svc.fit(train_x, train_y)
#     # Y_pred = svc.predict(X_test)
#     acc_svc = round(svc.score(train_x, train_y) * 100, 2)
#     print(acc_svc)
#
#     # KNN
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(train_x, train_y)
#     # Y_pred = knn.predict(X_test)
#     acc_knn = round(knn.score(train_x, train_y) * 100, 2)
#     print(acc_knn)
#
#     # Gaussian Naive Bayes
#
#     gaussian = GaussianNB()
#     gaussian.fit(train_x, train_y)
#     # Y_pred = gaussian.predict(X_test)
#     acc_gaussian = round(gaussian.score(train_x, train_y) * 100, 2)
#     print(acc_gaussian)


if __name__ == "__main__":

    """paramaters for running learning algorithms"""
    verbose = True
    # num threads always set to 1. Or, if using -1 this will hyper thread
    threads = 1
    seed = 1
    timings = {}

    """load in data"""
    ds1_details = {'data': ld.CreditDefaultData(verbose=verbose, seed=seed),
                   'name': 'credit_default',
                   'readable_name': 'Credit Default'}

    """define list to hold multiple data sets"""
    datasets = [ds1_details]

    """track the different experiments to conduct"""
    experiment_details = []
    for ds in datasets:
        """calls dict key: data which uses a loader function object"""
        data = ds['data']
        """
        each dataset has a class in loader.py
        each class inherits from parent: class DataLoader(ABC)
        """
        # call data load, build train/test sets, standardize the data
        data.load_and_process()
        data.build_train_test_split()
        data.scale_standard()
        # call this once to set up the experiment details
        experiment_details.append(experiments.ExperimentDetails(data,
                                                                ds['name'],
                                                                ds['readable_name'],
                                                                threads=threads,
                                                                seed=seed))

    DT_run = False
    ANN_run = False
    KNN_run = True
    SVM_run = False
    BOOSTING_run = False

    if DT_run:
        print("Start DT Experiment")
        run_experiment(experiment_details, experiments.DTExperiment, 'DT', verbose, timings)

    if ANN_run:
        print("Start ANN Experiment")
        run_experiment(experiment_details, experiments.ANNExperiment, 'ANN', verbose, timings)

    if KNN_run:
        print("Start KNN Experiment")
        run_experiment(experiment_details, experiments.KNNExperiment, 'KNN', verbose, timings)

    if SVM_run:
        print("Start DT Experiment")
        # TODO: change experiments.DTExperiment to the appropriate SVM
        run_experiment(experiment_details, experiments.DTExperiment, 'SVM', verbose, timings)

    if BOOSTING_run:
        print("Start DT Experiment")
        # TODO: change experiments.DTExperiment to the appropriate Boosting
        run_experiment(experiment_details, experiments.DTExperiment, 'Boosting', verbose, timings)
