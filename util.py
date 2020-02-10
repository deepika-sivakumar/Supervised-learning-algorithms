from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import datetime

def get_scores(estimator, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       scoring='accuracy',
                       train_sizes=train_sizes,
                       return_times=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    return train_scores_mean, test_scores_mean

def plot_learning_curve(estimator, title, X, y, graph_name, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       scoring='accuracy',
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation Test score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.legend(loc="best")
    generate_graph(graph_name+"lc", "Learning Curve", "Training set size", "Accuracy Score")

    # Plot Training size vs fit_times
    plt.plot(train_sizes, fit_times_mean, 'o-', label="Model Fit Time")
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    plt.legend(loc="best")
    generate_graph(graph_name+"sca", "Scalability of the model", "Training set size", "Fit Time(Seconds)")

    # Plot fit_time vs score
    plt.plot(fit_times_mean, test_scores_mean, 'o-', label="Validation Test Score")
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    # plt.annotate(text=train_sizes, xy=)
    plt.legend(loc="best")
    generate_graph(graph_name+"perf", "Performance of the model", "Fit Time", "Accuracy Score")

def plot_lc_nn(mlp_model, X, y, graph_name, train_sizes=np.linspace(.1, 1.0, 5)):
    mlp_train_score = []
    mlp_val_score = []
    fit_times = []
    for train_size in train_sizes:
        X_train, X_val_test, y_train, y_val_test = \
            train_test_split(X, y, train_size=train_size, random_state=7)
        start = datetime.datetime.now()
        mlp_model.fit(X_train, y_train)
        finish = datetime.datetime.now()
        y_train_predict = mlp_model.predict(X_train)
        y_val_test_predict = mlp_model.predict(X_val_test)
        mlp_train_score.append(accuracy_score(y_train, y_train_predict))
        mlp_val_score.append(accuracy_score(y_val_test, y_val_test_predict))
        fit_times.append((finish-start).total_seconds())
    # Plot learning curve
    plt.plot(train_sizes, mlp_train_score, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, mlp_val_score, 'o-', color="g",
             label="Validation Test score")
    plt.legend(loc="best")
    generate_graph(graph_name + "lc", "Learning Curve", "Training set size", "Accuracy Score")

    # Plot Training size vs fit_times
    plt.plot(train_sizes, fit_times, 'o-', label="Model Fit Time")
    plt.legend(loc="best")
    generate_graph(graph_name + "sca", "Scalability of the model", "Training set size", "Fit Time(Seconds)")

    # Plot fit_time vs score
    plt.plot(fit_times, mlp_val_score, 'o-', label="Validation Test Score")
    plt.legend(loc="best")
    generate_graph(graph_name + "perf", "Performance of the model", "Fit Time", "Accuracy Score")


def generate_graph(filename, title, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend()
    # plt.show()
    plt.savefig('graphs/'+ filename)
    plt.close()