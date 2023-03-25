import numpy as np
import numpy


# For linear regression, we are trying to find a response vector for a given feature vector
# using our continuous line-of-best-fit. LinReg is best used for getting inbetween data
# situated in gaps in discrete data.


class LinReg:
    def __init__(self):
        self.learning_rate = 0.001  # Small so our model is more stable
        self.weights = None
        self.bias = 0

    def fit_data(self, training, labels, iterations: int):
        items, shape = training.shape
        # Randomize weights
        self.weights = numpy.random.rand(shape)
        for i in range(iterations):
            # Uses linear regression formula
            aprx_y = numpy.dot(training, self.weights) + self.bias
            dWeights = (1 / items) * numpy.dot(training.T, aprx_y - labels)
            dBias = (1 / items) * numpy.sum(aprx_y - labels)
            # Gradient Decent
            self.weights -= self.learning_rate * dWeights
            self.bias -= self.learning_rate * dBias

    def predict_result(self, testing):
        aprx_y = numpy.dot(testing, self.weights) + self.bias
        return aprx_y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Training data from sklearn
    X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=4321)

    # Training data graphed using matplotlib
    figure = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], Y, color="r", marker="o", s=30)

    # Comparing my Linear Regression with the one in sklearn
    from sklearn.linear_model import LinearRegression

    SciLinReg = LinearRegression()
    SciLinReg.fit(X_train, Y_train)
    Y_predict = SciLinReg.predict(X_test)

    # My Linear Regression Model
    MyLinReg = LinReg()
    MyLinReg.fit_data(X_train, Y_train, iterations=3000)
    My_Y_predict = MyLinReg.predict_result(X_test)

    # My Model (Blue Line)
    plt.plot(X_test, My_Y_predict, color='b')
    # Sklearn Model (Black Line)
    plt.plot(X_test, Y_predict, color='k')
    plt.show()

    # Statistical comparison of my model
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    R_Squared = r2_score(y_true=Y_test, y_pred=Y_predict)
    Mean_Absolute_Error = mean_squared_error(y_true=Y_test, y_pred=Y_predict)
    Mean_Squared_Error = mean_absolute_error(y_true=Y_test, y_pred=Y_predict)
    My_R_Squared = r2_score(y_true=Y_test, y_pred=My_Y_predict)
    My_Mean_Absolute_Error = mean_squared_error(y_true=Y_test, y_pred=My_Y_predict)
    My_Mean_Squared_Error = mean_absolute_error(y_true=Y_test, y_pred=My_Y_predict)

    print(f"Sklearn: {R_Squared=}")
    print(f"My: {My_R_Squared=}")

    print(f"Sklearn: {Mean_Absolute_Error=}")
    print(f"My: {My_Mean_Absolute_Error=}")

    print(f"Sklearn: {Mean_Squared_Error=}")
    print(f"My: {My_Mean_Squared_Error=}")

