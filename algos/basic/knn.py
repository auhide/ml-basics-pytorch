from typing import Union

import torch

from base import Model


class KNearestNeighborsClassifier(Model):

    def __init__(self, k=5, metric="l2"):
        assert metric in ["l1", "l2"], "`metric` should be either 'l1' or 'l2' for Manhattan norm and Euclidean norm respectively."

        self.k = k
        self.metric = metric

    def fit(self, X: torch.Tensor, y: torch.LongTensor):
        # As opposed to the other algorithms, here we don't have traditional
        # weights. The training dataset is used to calculate distances between
        # it and each prediction.
        self.X, self.y = X, y

    def predict(self, X: torch.Tensor) -> torch.LongTensor:
        predictions = []
        
        for vector in X:
            distances = self._calc_norm(vector)
            # The top K smallest distances between the input tensor `vector` and
            # the training dataset tensors.
            topk_ids = distances.topk(self.k).indices
            # Select the classes of the smallest distances.
            topk_classes = torch.index_select(
                self.y,
                dim=0,
                index=topk_ids
            ).flatten()

            prediction = int(topk_classes.mode(dim=0).values)
            predictions.append(prediction)

        return torch.LongTensor(predictions)

    def evaluate(self, X: torch.Tensor, y: torch.LongTensor) -> float:
        y_pred = self.predict(X)
        # Since it is a classifier we are going to calculate the accuracy.
        y = y.flatten()
        accuracy = self._accuracy(y_pred, y)
        
        return accuracy

    def _calc_norm(self, X: torch.Tensor) -> torch.Tensor:
        metric_type = "fro" if self.metric == "l2" else "1"
        distances = (X - self.X).norm(p=metric_type, dim=1)

        return distances
    
    def _accuracy(self, y_pred: torch.LongTensor, y: torch.LongTensor):
        return (y_pred == y).int().count_nonzero() / len(y)


if __name__ == "__main__":
    X = torch.randint(low=1, high=5, size=(10, 5)).float()
    y = torch.randint(low=0, high=2, size=(10, 1))

    knn = KNearestNeighborsClassifier(k=5)
    knn.fit(X, y)
    
    predictions = knn.predict(X)
    accuracy = knn.evaluate(X, y)
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"Accuracy: {accuracy}") 
