import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    # number of samples and features
    n_samples, n_features = X.shape
    
    # initialize weights and bias
    w = np.zeros(n_features)
    b = 0
    
    for i in range(steps):
        
        # linear equation
        z = np.dot(X, w) + b
        
        # prediction
        y_hat = _sigmoid(z)
        
        # gradients
        dw = np.dot(X.T, (y_hat - y)) / n_samples
        db = np.sum(y_hat - y) / n_samples
        
        # update weights
        w = w - lr * dw
        b = b - lr * db
        
    return w, b