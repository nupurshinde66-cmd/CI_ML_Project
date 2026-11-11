
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model():
    X = [[1], [2], [3], [4], [5], [6], [7], [8]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    return model, accuracy
