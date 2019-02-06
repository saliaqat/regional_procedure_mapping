from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel():

    def __init__(self, penalty='l2', solver='newton-cg'):
        self.model = LogisticRegression(penalty=penalty, solver=solver)

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

