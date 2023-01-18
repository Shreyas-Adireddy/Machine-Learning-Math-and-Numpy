import numpy
import math
import collections
# HOW DOES IT WORK?
# When a new point is being tested in your training dataset, the model checks how close the
# current point is to the K nearest neighbors (KNN) usually, the 3 nearest neighbors.
# HOW TO USE?
# K is usually selected to be the square root of the total data points rounded to an odd number
# It is rounded to an odd number because usually there are 2 classes and 3 an even K value could
# lead to a situation where our testing point is equidistant from our even K nearest neighbors.
# To avoid this, an odd K value is preferred.


def sort_by_index(seq):
    return [index for index, value in sorted(enumerate(seq), key=lambda x: x[1])]


class KNN_model:
    def __init__(self, K):
        self.training = None
        self.labels = None
        self.K = K

    def assign(self, training, labels):
        self.training = training
        self.labels = labels

    def make_prediction_array(self, testing):
        prediction = []
        for value in testing:
            prediction.append(self.make_predict_individual(value))
        return prediction

    def make_predict_individual(self, value):
        # Look at nearest neighbors using Manhattan distance formula
        def distance(point1, point2):
            math.sqrt(math.fsum((point1-point2)**2))
        all_distances = [distance(value, x_value) for x_value in self.training]
        # Get the indices of the array if it was sorted. Doesn't sort the array.
        all_k_nearest_neighbors_idx = sort_by_index(all_distances)[:self.K]
        k_nearest_neighbors_labels = [self.labels[i] for i in all_k_nearest_neighbors_idx]
        # Gets the best label/most occurring
        best_label = collections.Counter(k_nearest_neighbors_labels).most_common(1)
        return best_label[0][0]




