import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import util
from sklearn.neural_network import MLPClassifier

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
    # Plot learning curves before tuning with default hidden layers
    mlp_model = MLPClassifier(hidden_layer_sizes=(10,10,10),random_state=random_seed)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    util.plot_lc_nn(mlp_model=mlp_model, X=X, y=y, train_sizes=train_sizes, graph_name='nn/nn_wine_')


    # Hyperparameter tuning, hidden layer size
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    hidden_layer_sizes = [10, 30, 50, 80, 100]
    train_score = []
    test_score = []
    for i in hidden_layer_sizes:
        mlp_model = MLPClassifier(hidden_layer_sizes=(i,i,i),random_state=random_seed)
        mlp_model.fit(X=X_train, y=y_train)
        y_train_predict = mlp_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = mlp_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_layers = pd.DataFrame({
        'Hidden layer sizes': hidden_layer_sizes,
        'train score': train_score,
        'validation score': test_score
    })
    print('Hidden layers**************')
    print(df_layers)

    # Plot Max depth
    plt.plot(hidden_layer_sizes, train_score, 'o-', color="r",
              label="Training score")
    plt.plot(hidden_layer_sizes, test_score, 'o-', color="g",
              label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("nn/nn_wine_layers", "Hidden layer sizes Vs Accuracy",
                        "Hidden layer sizes", "Accuracy Score")

    # Choosing layer size = 30
    # Decision Tree after pruning/tuning
    mlp_model = MLPClassifier(hidden_layer_sizes=(30, 30, 30), random_state=random_seed)
    util.plot_lc_nn(mlp_model=mlp_model, X=X, y=y, train_sizes=train_sizes, graph_name='nn/nn_wine_tuned_')

    # Final Model Accuracy against test set we kept aside, with max_depth = 11
    mlp_model = MLPClassifier(hidden_layer_sizes=(30, 30, 30), random_state=random_seed)
    mlp_model.fit(X,y)
    y_predict = mlp_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("MLPClassifier - Wine Dataset - Final Accuracy score on the test set: ", final_accuracy)

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
    # Plot learning curves before tuning with default hidden layers
    mlp_model = MLPClassifier(hidden_layer_sizes=(1),random_state=random_seed)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    util.plot_lc_nn(mlp_model=mlp_model, X=X, y=y, train_sizes=train_sizes, graph_name='nn/nn_htru_')

    # Hyperparameter tuning, hidden layer size
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    hidden_layer_sizes = [1, 3, 5, 7, 10]
    train_score = []
    test_score = []
    for i in hidden_layer_sizes:
        mlp_model = MLPClassifier(hidden_layer_sizes=(i),random_state=random_seed)
        mlp_model.fit(X=X_train, y=y_train)
        y_train_predict = mlp_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = mlp_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_layers = pd.DataFrame({
        'Hidden layer sizes': hidden_layer_sizes,
        'train score': train_score,
        'validation score': test_score
    })
    print('Hidden layers**************')
    print(df_layers)

    # Plot Max depth
    plt.plot(hidden_layer_sizes, train_score, 'o-', color="r",
              label="Training score")
    plt.plot(hidden_layer_sizes, test_score, 'o-', color="g",
              label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("nn/nn_htru_layers", "Hidden layer sizes Vs Accuracy",
                        "Hidden layer sizes", "Accuracy Score")


    # Choosing layer size = 3
    # Decision Tree after pruning/tuning
    mlp_model = MLPClassifier(hidden_layer_sizes=(3), random_state=random_seed)
    util.plot_lc_nn(mlp_model=mlp_model, X=X, y=y, train_sizes=train_sizes, graph_name='nn/nn_htru_tuned_')

    # Final Model Accuracy against test set we kept aside, with max_depth = 11
    mlp_model = MLPClassifier(hidden_layer_sizes=(3), random_state=random_seed)
    mlp_model.fit(X,y)
    y_predict = mlp_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("MLPClassifier - HTRU_2 Dataset - Final Accuracy score on the test set: ", final_accuracy)

if __name__=="__main__":
    wine_dataset()
    pulsar_dataset()

