import torch

from base import Model


class SupportVectorMachine(Model):

    def __init__(self):
        self.w = None
        self.lambda_ = None

    def fit(self, X, y, iters=500, l_rate=1e-1, lambda_=0.1):
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

        return X @ self.w

    def evaluate(self, X, y):
        X = self._bias_reshape_X(X)

        with torch.no_grad():
            y_pred = X @ self.w
            loss = self._loss(y_pred, y, self.lambda_)

        return loss.item()

    def _loss(self, y_pred, y, lambda_):
        hinge = self._hinge_loss(y_pred, y).mean()
        l2_regularization = (self.w ** 2).sum()

        return lambda_ * hinge + l2_regularization

    def _hinge_loss(self, y_pred, y):
        indicator_func = y_pred * y
        loss = 1 - indicator_func
        loss[loss < 0] = 0

        return loss

    def _bias_reshape_X(self, X: torch.Tensor):
        batch_size = X.shape[0]
        X = torch.cat([X, torch.ones((batch_size, 1))], dim=-1)

        return X


if __name__ == "__main__":
    X = torch.randn(size=(10, 5))
    y = torch.Tensor([-1] * 5 + [1] * 5)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)

    svm = SupportVectorMachine()
    svm.fit(X, y)

    y_pred = svm.predict(X)
    print("y_pred.shape:", y_pred.shape)
