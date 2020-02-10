import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
    KNN_model = KNeighborsClassifier(n_neighbors=3)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=KNN_model, title='Learning Curve - KNN', X=X, y=y,
                             cv=3, train_sizes = train_sizes, graph_name= 'knn/knn_wine_')

    # Tuning the KNN model by the n_neighbours parameter
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    k_neighbours = range(1,31)
    train_score = []
    test_score = []
    for k in k_neighbours:
        KNN_model = KNeighborsClassifier(n_neighbors=k)
        KNN_model.fit(X=X_train, y=y_train)
        y_train_predict = KNN_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = KNN_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_neighbours = pd.DataFrame({
        'No. neighbours': k_neighbours,
        'train score': train_score,
        'test score': test_score
    })
    print('K neighbours**************')
    print(df_neighbours)

    # Plot Max depth
    plt.plot(k_neighbours, train_score, 'o-', color="r",
             label="Training score")
    plt.plot(k_neighbours, test_score, 'o-', color="g",
             label="Test score")
    plt.legend(loc="best")
    util.generate_graph("knn/knn_wine_nei", "K neighbours Vs Accuracy",
                        "K neighbours", "Accuracy Score")

    # At k = 14, we get a good meeting of train and validation scores.
    # KNN model after tuning
    KNN_model = KNeighborsClassifier(n_neighbors=14)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=KNN_model, title='Learning Curve - KNN', X=X, y=y,
                             cv=3, train_sizes=train_sizes, graph_name='knn/knn_wine_tuned_')
    # Final Model Accuracy against test set we kept aside, with k = 14
    KNN_model = KNeighborsClassifier(n_neighbors=14)
    KNN_model.fit(X, y)
    y_predict = KNN_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("KNeighborsClassifier - Wine Dataset - Final Accuracy score on the test set: ", final_accuracy)

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
    KNN_model = KNeighborsClassifier(n_neighbors=3)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=KNN_model, title='Learning Curve - KNN', X=X, y=y,
                             cv=3, train_sizes = train_sizes, graph_name= 'knn/knn_htru_')

    # Tuning the KNN model by the n_neighbours parameter
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    k_neighbours = range(1,31)
    train_score = []
    test_score = []
    for k in k_neighbours:
        KNN_model = KNeighborsClassifier(n_neighbors=k)
        KNN_model.fit(X=X_train, y=y_train)
        y_train_predict = KNN_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = KNN_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_neighbours = pd.DataFrame({
        'No. neighbours': k_neighbours,
        'train score': train_score,
        'test score': test_score
    })
    print('K neighbours**************')
    print(df_neighbours)

    # Plot Max depth
    plt.plot(k_neighbours, train_score, 'o-', color="r",
             label="Training score")
    plt.plot(k_neighbours, test_score, 'o-', color="g",
             label="Test score")
    plt.legend(loc="best")
    util.generate_graph("knn/knn_htru_nei", "K neighbours Vs Accuracy",
                        "K neighbours", "Accuracy Score")


    # At k = 2, we get a good meeting of train and validation scores.
    # KNN model after tuning
    KNN_model = KNeighborsClassifier(n_neighbors=2)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=KNN_model, title='Learning Curve - KNN', X=X, y=y,
                             cv=3, train_sizes=train_sizes, graph_name='knn/knn_htru_tuned_')
    # Final Model Accuracy against test set we kept aside, with k = 14
    KNN_model = KNeighborsClassifier(n_neighbors=2)
    KNN_model.fit(X, y)
    y_predict = KNN_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("KNeighborsClassifier - HTRU_2 Dataset - Final Accuracy score on the test set: ", final_accuracy)

if __name__=="__main__":
    wine_dataset()
    pulsar_dataset()
