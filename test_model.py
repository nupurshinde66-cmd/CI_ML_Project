
from ml_model import train_model

def test_model_training():
    model, accuracy = train_model()
    assert model is not None

def test_model_accuracy():
    model, accuracy = train_model()
    assert accuracy >= 0.80

def test_model_prediction():
    model, accuracy = train_model()
    prediction = model.predict([[3]])
    assert prediction[0] in [0, 1]
