import numpy as np


class LogisticalRegression:
    def __init__(self, learning_rate: int = 0.001, iterations: int = 10000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        total, shape = X.shape
        # Randomize weights
        self.weights = np.zeros(shape)

        for _ in range(self.iterations):
            line = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(line)
            # Gradient of weights
            # (1/N)*xi*(y_hat-y)
            dw = (1 / total) * np.dot(X.T, (y_hat - y))
            # Gradient of weights
            # (1/N)*(y_hat-y)
            db = (1 / total) * np.sum(y_hat - y)
            # Update Params
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        y = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(y)
        results = np.zeros_like(y_hat)
        results[y_hat >= 0.5] = 1
        return y_hat


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    digits = load_iris()
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
    plt.show()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    Sklearn_Reg = LogisticRegression()
    Our_Reg = LogisticalRegression()

    Sklearn_Reg.fit(x_train, y_train)
    Our_Reg.fit(x_train, y_train)

    predictions1 = Sklearn_Reg.predict(x_test)
    predictions2 = Our_Reg.predict(x_test)

    print("Sklearn Accuracy: ", accuracy_score(predictions1, y_test))
    print("Our Model Accuracy: ", accuracy_score(predictions2, y_test))

    cm = confusion_matrix(y_test, predictions1)
    cm2 = confusion_matrix(y_test, predictions2)

    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 0, 2])

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Define the labels for the classes
    classes = ['Class 0', 'Class 1', 'Class 2']

    # Display the confusion matrix as a heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over the data and create annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")

    fig, ax2 = plt.subplots()
    im = ax2.imshow(cm2, interpolation='nearest',cmap="autumn")
    ax2.figure.colorbar(im, ax=ax2)
    ax2.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over the data and create annotations
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm2[i, j] > cm.max() / 2. else "black")
    plt.show()