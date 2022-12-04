import torch

from base import Model


class SupportVectorMachine(Model):

    def __init__(self):
        self.w = None
        self.lambda_ = None

    def fit(self, X, y, iters=500, l_rate=1e-1, lambda_=2):
        # Adding a bias column (a column of 1s) to the X tensor.
        X = self._bias_reshape_X(X)
        N_FEATURES = X.shape[-1]

        self.lambda_ = lambda_
        self.w = torch.randn(size=(N_FEATURES, 1), requires_grad=True)

        for i in range(iters):
            y_pred = X @ self.w
            loss = self._loss(y_pred, y, lambda_)
            loss.backward()

            with torch.no_grad():
                self.w -= l_rate * self.w.grad
                self.w.grad.zero_()

            if i % 20 == 0:
                print(f"Loss: {loss.item():.2f}")

    def predict(self, X):
        X = self._bias_reshape_X(X)
        print(X.shape)

        return X @ self.w

    def evaluate(self, X, y):
        print(X.shape, self.w.shape)
        X = self._bias_reshape_X(X)

        with torch.no_grad():
            y_pred = X @ self.w
            loss = self._loss(y_pred, y, self.lambda_)

        return loss.item()

    def _loss(self, y_pred, y, lambda_):
        BATCH_SIZE = y.shape[0]
        factor = 1 / BATCH_SIZE

        # Computing loss for the soft-margin SVM classifier.
        loss = (
            factor * torch.sum(
                torch.max(torch.Tensor(0), 1 - y @ (y_pred))
            )
        ) + lambda_ * torch.linalg.norm(self.w)

        return loss

    def _bias_reshape_X(self, X: torch.Tensor):
        batch_size = X.shape[0]
        X = torch.cat([X, torch.ones((batch_size, 1))], dim=-1)

        return X


if __name__ == "__main__":
    X = torch.randn(size=(10, 5))
    y = torch.Tensor([-1] * 5 + [1] * 5)
    print(X.shape)
    print(y.shape)

    svm = SupportVectorMachine()
    svm.fit(X, y)
