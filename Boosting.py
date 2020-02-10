from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import util

def wine_dataset():
    random_seed = 7
    df = pd.read_csv('datasets/winequality-white.csv', sep=';')
    df = df.dropna()
    print('data size***********', df.shape)
    # Let us keep aside data for final testing, since we are going to employ cross-validation
    data_X = df.iloc[:, :-1]
    data_y = df.iloc[:, -1]
    X, X_test, y, y_test = train_test_split(data_X, data_y, train_size=0.8, random_state=random_seed)
    # We will use X,y for tuning the model
    boost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(random_state=random_seed),
        n_estimators=10)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=boost_model, title='Learning Curve - Ada Boost Classifier',
                             X=X, y=y, cv=3, train_sizes = train_sizes, graph_name= 'boost/boost_wine_')


    # Let's choose training set size 0.8, since dataset seems almost evenly distributed
    # Tuning no of estimators
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    no_estimators = [10, 100, 150, 200]
    train_score = []
    test_score = []
    for i in no_estimators:
        boost_model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(random_state=random_seed),
            n_estimators=i, random_state=random_seed)
        boost_model.fit(X=X_train, y=y_train)
        y_train_predict = boost_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = boost_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_depth = pd.DataFrame({
        'No Estimators': no_estimators,
        'train score': train_score,
        'validation score': test_score
    })
    print('No Estimators**************')
    print(df_depth)

    # Plot Max depth
    plt.plot(no_estimators, train_score, 'o-', color="r",
              label="Training score")
    plt.plot(no_estimators, test_score, 'o-', color="g",
              label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("boost/boost_wine_estimators", "No of Estimators Vs Accuracy",
                        "No Estimators", "Accuracy Score")

    # Let us take no_estimators = 10
    max_depths = range(1,31)
    train_score = []
    test_score = []
    for max_depth in max_depths:
        boost_model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(random_state=random_seed, max_depth=max_depth),
            n_estimators=10, random_state=random_seed)
        boost_model.fit(X=X_train, y=y_train)
        y_train_predict = boost_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = boost_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_depth = pd.DataFrame({
        'max_depths': max_depths,
        'train score': train_score,
        'validation score': test_score
    })
    print('Max depth**************')
    print(df_depth)

    # Plot Max depth
    plt.plot(max_depths, train_score, 'o-', color="r",
             label="Training score")
    plt.plot(max_depths, test_score, 'o-', color="g",
             label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("boost/boost_wine_depth", "Max Depth Vs Accuracy",
                        "Max depth", "Accuracy Score")


    # At max_depth = 8, test score = 0.585459, train = 0.843012
    # Avoid too much overfitting
    # Decision Tree after pruning/tuning
    boost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=8, random_state=random_seed),
        n_estimators=10)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=boost_model, title='Learning Curve - Ada Boost Classifier',
                             X=X, y=y, cv=3, train_sizes=train_sizes, graph_name='boost/boost_wine_pruned_')

    # Final Model Accuracy against test set we kept aside, with max_depth = 11
    boost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=8, random_state=random_seed),
        n_estimators=10)
    boost_model.fit(X,y)
    y_predict = boost_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("AdaBoostClassifier - Wine Dataset - Final Accuracy score on the test set: ", final_accuracy)

def pulsar_dataset():
    random_seed = 7
    df = pd.read_csv('datasets/HTRU_2.csv')
    df = df.dropna()
    print('data size***********', df.shape)
    # Let us keep aside data for final testing, since we are going to employ cross-validation
    data_X = df.iloc[:, :-1]
    data_y = df.iloc[:, -1]
    X, X_test, y, y_test = train_test_split(data_X, data_y, train_size=0.8, random_state=random_seed)
    # We will use X,y for tuning the model
    boost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(random_state=random_seed),
        n_estimators=10)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=boost_model, title='Learning Curve - Ada Boost Classifier',
                             X=X, y=y, cv=3, train_sizes = train_sizes, graph_name= 'boost/boost_htru_')


    # Let's choose training set size 0.8, since dataset seems almost evenly distributed
    # Tuning no of estimators
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    no_estimators = [10, 100, 150, 200]
    train_score = []
    test_score = []
    for i in no_estimators:
        boost_model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(random_state=random_seed),
            n_estimators=i, random_state=random_seed)
        boost_model.fit(X=X_train, y=y_train)
        y_train_predict = boost_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = boost_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_depth = pd.DataFrame({
        'No Estimators': no_estimators,
        'train score': train_score,
        'validation score': test_score
    })
    print('No Estimators**************')
    print(df_depth)

    # Plot Max depth
    plt.plot(no_estimators, train_score, 'o-', color="r",
              label="Training score")
    plt.plot(no_estimators, test_score, 'o-', color="g",
              label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("boost/boost_htru_estimators", "No of Estimators Vs Accuracy",
                        "No Estimators", "Accuracy Score")

    # Let us take no_estimators = 10
    max_depths = range(1,31)
    train_score = []
    test_score = []
    for max_depth in max_depths:
        boost_model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(random_state=random_seed, max_depth=max_depth),
            n_estimators=10, random_state=random_seed)
        boost_model.fit(X=X_train, y=y_train)
        y_train_predict = boost_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = boost_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_depth = pd.DataFrame({
        'max_depths': max_depths,
        'train score': train_score,
        'validation score': test_score
    })
    print('Max depth**************')
    print(df_depth)

    # Plot Max depth
    plt.plot(max_depths, train_score, 'o-', color="r",
             label="Training score")
    plt.plot(max_depths, test_score, 'o-', color="g",
             label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("boost/boost_htru_depth", "Max Depth Vs Accuracy",
                        "Max depth", "Accuracy Score")


    # At max_depth = 1, test score = 0.976955, train = 0.978086, not much difference increasing depth
    # so going with a very simple tree
    # Avoid too much overfitting
    # Decision Tree after pruning/tuning
    boost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=random_seed),
        n_estimators=10)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=boost_model, title='Learning Curve - Ada Boost Classifier',
                             X=X, y=y, cv=3, train_sizes=train_sizes, graph_name='boost/boost_htru_pruned_')

    # Final Model Accuracy against test set we kept aside, with max_depth = 1
    boost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=random_seed),
        n_estimators=10)
    boost_model.fit(X,y)
    y_predict = boost_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("AdaBoostClassifier - HTRU_2 Dataset - Final Accuracy score on the test set: ", final_accuracy)

if __name__=="__main__":
    wine_dataset()
    pulsar_dataset()

