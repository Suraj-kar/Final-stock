import numpy as np
import joblib

class DataScaler:
    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds
    
    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
    
    def transform(self, X):
        return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def save(self, file_path):
        joblib.dump({'means': self.means, 'stds': self.stds}, file_path)
    
    @classmethod
    def load(cls, file_path):
        data = joblib.load(file_path)
        return cls(means=data['means'], stds=data['stds'])



class MultiLinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, weights=None, bias=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = weights
        self.bias = bias
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for i in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if i % 100 == 0:
                mse = np.mean((y - y_predicted) ** 2)
                print(f"Iteration {i}: MSE = {mse:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def save(self, file_path):
        joblib.dump({'weights': self.weights, 'bias': self.bias}, file_path)
    
    @classmethod
    def load(cls, file_path):
        data = joblib.load(file_path)
        return cls(weights=data['weights'], bias=data['bias'])
    
    
    