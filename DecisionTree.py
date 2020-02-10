import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
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
    DT_model = tree.DecisionTreeClassifier(random_state=random_seed)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=DT_model, title='Learning Curve - Decision Trees', X=X, y=y,
                             cv=3, train_sizes = train_sizes, graph_name= 'dt/dt_wine_')
    # Let's choose training set size 0.8, since dataset seems almost evenly distributed
    # Pruning
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_score = []
    test_score = []
    for max_depth in max_depths:
        DT_model = tree.DecisionTreeClassifier(max_depth = max_depth, random_state = random_seed)
        DT_model.fit(X=X_train, y=y_train)
        y_train_predict = DT_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = DT_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_depth = pd.DataFrame({
        'max depth': max_depths,
        'train score': train_score,
        'validation score': test_score
    })
    print('max depth**************')
    print(df_depth)

    # Plot Max depth
    plt.plot(max_depths, train_score, 'o-', color="r",
              label="Training score")
    plt.plot(max_depths, test_score, 'o-', color="g",
              label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("dt/dt_wine_max_depths", "Decision Tree Depths Vs Accuracy",
                        "Max Tree Depth", "Accuracy Score")
    # At max_depth =11, test score = 0.6020408163265306, train = 0.7937723328228689
    # max edpth = 19 , test score = 0.6306122448979592, train = 0.9913221031138336
    # Let is learn , at 1, my train and test scores are almost same.
    # choose max_depth = 19
    # Decision Tree after pruning/tuning
    DT_model = tree.DecisionTreeClassifier(max_depth=11, random_state=random_seed)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=DT_model, title='Learning Curve - Decision Trees', X=X, y=y,
                             cv=3, train_sizes=train_sizes, graph_name= 'dt/dt_wine_pruned_')
    # Final Model Accuracy against test set we kept aside, with max_depth = 11
    DT_model = tree.DecisionTreeClassifier(max_depth=11, random_state=random_seed)
    DT_model.fit(X,y)
    y_predict = DT_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("DecisionTreeClassifier - Wine Dataset - Final Accuracy score on the test set: ", final_accuracy)

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
    DT_model = tree.DecisionTreeClassifier(random_state=random_seed)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=DT_model, title='Learning Curve - Decision Trees', X=X, y=y,
                             cv=3, train_sizes = train_sizes, graph_name= 'dt/dt_htru_')
    # Let's choose training set size 0.8, since dataset seems almost evenly distributed
    # Pruning
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_score = []
    test_score = []
    for max_depth in max_depths:
        DT_model = tree.DecisionTreeClassifier(max_depth = max_depth, random_state = random_seed)
        DT_model.fit(X=X_train, y=y_train)
        y_train_predict = DT_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = DT_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_depth = pd.DataFrame({
        'max depth': max_depths,
        'train score': train_score,
        'validation score': test_score
    })
    print('max depth**************')
    print(df_depth)

    # Plot Max depth
    plt.plot(max_depths, train_score, 'o-', color="r",
              label="Training score")
    plt.plot(max_depths, test_score, 'o-', color="g",
              label="Validation score")
    plt.legend(loc="best")
    util.generate_graph("dt/dt_htru_max_depths", "Decision Tree Depths Vs Accuracy",
                        "Max Tree Depth", "Accuracy Score")

    # choose max_depth = 1
    # Decision Tree after pruning/tuning
    DT_model = tree.DecisionTreeClassifier(max_depth=1, random_state=random_seed)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=DT_model, title='Learning Curve - Decision Trees', X=X, y=y,
                             cv=3, train_sizes=train_sizes, graph_name= 'dt/dt_htru_pruned_')
    # Final Model Accuracy against test set we kept aside, with max_depth = 11
    DT_model = tree.DecisionTreeClassifier(max_depth=1, random_state=random_seed)
    DT_model.fit(X,y)
    y_predict = DT_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("DecisionTreeClassifier - HTRU_2 Dataset - Final Accuracy score on the test set: ", final_accuracy)

if __name__=="__main__":
    wine_dataset()
    pulsar_dataset()
