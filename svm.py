import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import util
from sklearn import svm
from sklearn.decomposition import PCA

"""
def pca():
    pca = PCA(n_components=2).fit(X_train)
    pca_2d = pca.transform(X_train)
"""

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
    svm_model = svm.SVC(kernel='poly',random_state=random_seed)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before tuning
    util.plot_learning_curve(estimator=svm_model, title='Learning Curve - Decision Trees', X=X, y=y,
                             cv=3, train_sizes = train_sizes, graph_name= 'svm/svm_wine_poly_')

    # Swapping kernels in the SVM model
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    kernels = ['poly', 'rbf']
    train_score = []
    test_score = []
    for kernel in kernels:
        svm_model = svm.SVC(kernel = kernel,random_state=random_seed)
        svm_model.fit(X=X_train, y=y_train)
        y_train_predict = svm_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = svm_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_kernels = pd.DataFrame({
        'SVM kernel': kernels,
        'train score': train_score,
        'test score': test_score
    })
    print('SVM Kernels**************')
    print(df_kernels)

    # Plot Kernels
    plt.plot(kernels, train_score, 'o-', color="r",
             label="Training score")
    plt.plot(kernels, test_score, 'o-', color="g",
             label="Test score")
    plt.legend(loc="best")
    util.generate_graph("svm/svm_wine_kernels", "SVM Kernels Vs Accuracy",
                        "SVM Kernels", "Accuracy Score")

    # Accuracy score is more or less same for both kernels
    # But the performance (fit time) for rbf(0.2s) is lesser compared to poly(0.4s)
    svm_model = svm.SVC(kernel='rbf', random_state=random_seed)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=svm_model, title='Learning Curve - SVM', X=X, y=y,
                             cv=3, train_sizes=train_sizes, graph_name='svm/svm_wine_rbf_')

    # Final Model Accuracy against test set we kept aside, with kernel = rbf
    svm_model = svm.SVC(kernel='rbf', random_state=random_seed)
    svm_model.fit(X, y)
    y_predict = svm_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("SVC - Wine Dataset - Final Accuracy score on the test set: ", final_accuracy)

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
    svm_model = svm.SVC(kernel='linear',random_state=random_seed)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Plot learning curves before tuning
    util.plot_learning_curve(estimator=svm_model, title='Learning Curve - Decision Trees', X=X, y=y,
                             cv=3, train_sizes = train_sizes, graph_name= 'svm/svm_htru_linear_')

    # Swapping kernels in the SVM model
    X_train, X_val_test, y_train, y_val_test = \
        train_test_split(X, y, train_size=0.8, random_state=random_seed)
    kernels = ['linear', 'rbf']
    train_score = []
    test_score = []
    for kernel in kernels:
        svm_model = svm.SVC(kernel = kernel,random_state=random_seed)
        svm_model.fit(X=X_train, y=y_train)
        y_train_predict = svm_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_score.append(train_accuracy)

        y_val_test_predict = svm_model.predict(X_val_test)
        test_accuracy = accuracy_score(y_val_test, y_val_test_predict)
        test_score.append(test_accuracy)

    df_kernels = pd.DataFrame({
        'SVM kernel': kernels,
        'train score': train_score,
        'test score': test_score
    })
    print('SVM Kernels**************')
    print(df_kernels)

    # Plot Kernels
    plt.plot(kernels, train_score, 'o-', color="r",
             label="Training score")
    plt.plot(kernels, test_score, 'o-', color="g",
             label="Test score")
    plt.legend(loc="best")
    util.generate_graph("svm/svm_htru_kernels", "SVM Kernels Vs Accuracy",
                        "SVM Kernels", "Accuracy Score")

    # Accuracy score is more or less same for both kernels
    # But the performance (fit time) for rbf(0.2s) is lesser compared to linear(8s)
    svm_model = svm.SVC(kernel='rbf', random_state=random_seed)
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Plot learning curves before pruning
    util.plot_learning_curve(estimator=svm_model, title='Learning Curve - SVM', X=X, y=y,
                             cv=3, train_sizes=train_sizes, graph_name='svm/svm_htru_rbf_')

    # Final Model Accuracy against test set we kept aside, with kernel = rbf
    svm_model = svm.SVC(kernel='rbf', random_state=random_seed)
    svm_model.fit(X, y)
    y_predict = svm_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_predict)
    print("SVC - HTRU_2 Dataset - Final Accuracy score on the test set: ", final_accuracy)

if __name__=="__main__":
    wine_dataset()
    pulsar_dataset()