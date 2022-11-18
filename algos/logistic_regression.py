from tqdm import tqdm
import torch

from base import Model


class LogisticRegression(Model):

    def __init__(self):
        self.w = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, l_rate=1e-2, iters=100):
        # Defining the weights with random normally distributed values
        self.w = torch.randn(size=(X.shape[1], 1), requires_grad=True)

        # Training for 'iters' iterations
        for _ in tqdm(range(iters)):
            loss = self._log_likelihood_loss(self.predict(X), y)
            loss.backward()

            with torch.no_grad():
                self.w -= l_rate * self.w.grad
                self.w.grad.zero_()

    def predict(self, X: torch.Tensor):
        return self._sigmoid(X @ self.w)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            y_pred = self.predict(X)
        
        return self._log_likelihood_loss(y_pred, y)

    def _sigmoid(self, x: torch.Tensor):
        return 1 / (1 + torch.exp(x))

    def _log_likelihood_loss(self, y_pred, y):
        # We are going to divide scale_factor, which is equal to 
        # the number of samples/experiments.
        scale_factor = 1 / y_pred.shape[0]
        loss = torch.sum(
            (y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        ) / scale_factor

        return loss


if __name__ == "__main__":
    X = torch.randn(size=(10, 5))
    y = torch.cat([
        torch.zeros(size=(5, 1)), 
        torch.ones(size=(5, 1))
    ])

    regressor = LogisticRegression()
    regressor.fit(X, y)
