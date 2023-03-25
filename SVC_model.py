import numpy as np
import random


def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


class SVC:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.lambda_ = lambda_param  # tradeoff
        self.iterations = n_iters

    def fit_data(self, training, labels):
        sample, feat = training.shape

        # Changing all the labels to either -1 or 1
        labels = np.where(labels <= 0, -1, 1)
        self.weights = np.random.rand(feat)
        self.bias = random.uniform(-1, 1)

        for _ in range(self.iterations):
            for ind, val in enumerate(training):
                # yi(xi*w - b)≥1
                if labels[ind] * (np.dot(self.weights, val) - self.bias) >= 1:
                    # w = w + α* (2λw - yixi)
                    self.weights -= self.learning_rate * (2 * self.lambda_ * self.weights)
                else:
                    # w = w + α* (2λw - yixi)
                    self.weights -= self.learning_rate * (2 * self.lambda_ * self.weights - np.dot(val, labels[ind]))
                    # b = b - α* (yi)
                    self.bias -= self.learning_rate * labels[ind]

    def predict(self, testing):
        prediction = np.dot(testing, self.weights)
        return [1 if val > 0 else -1 for val in prediction]

    def visualize_svc(self):
        fig = plt.figure()
        plt.title("Our Implementation")
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

        x0_1 = np.amin(X_train[:, 0])
        x0_2 = np.amax(X_train[:, 0])

        x1_1 = get_hyperplane_value(x0_1, my_model.weights, my_model.bias, 0)
        x1_2 = get_hyperplane_value(x0_2, my_model.weights, my_model.bias, 0)

        x1_1_m = get_hyperplane_value(x0_1, my_model.weights, my_model.bias, -1)
        x1_2_m = get_hyperplane_value(x0_2, my_model.weights, my_model.bias, -1)

        x1_1_p = get_hyperplane_value(x0_1, my_model.weights, my_model.bias, 1)
        x1_2_p = get_hyperplane_value(x0_2, my_model.weights, my_model.bias, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X_train[:, 1])
        x1_max = np.amax(X_train[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()


if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.model_selection import train_test_split

    # Generate Data
    inputs, targets = datasets.make_blobs(n_samples=1000, centers=[(0, 0), (5, 5)], n_features=2, cluster_std=1.5)
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=1 / 3, random_state=100)

    plt.title('Sklearn Model')
    plt.xlabel('X_')
    plt.ylabel('y_')

    # Our Implementation
    my_model = SVC()
    my_model.fit_data(training=X_train, labels=y_train)

    # Sklearn Implementation
    sk_model = svm.SVC(kernel='linear')
    sk_model.fit(X_train, y_train)

    # Visualizing support vectors for sklearn model
    ax = plt.gca()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = sk_model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.show()

    my_model.visualize_svc()
