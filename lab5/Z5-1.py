import ssl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
import numpy as np


ssl._create_default_https_context = ssl._create_unverified_context

(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

my_model = load_model('fashion_mnist_model.keras')
cnn_accuracy = my_model.evaluate(X_test, y_test)

extractor = Sequential(my_model.layers[:-2])
extractor.compile()

X_train_extracted = extractor.predict(X_train)
X_test_extracted = extractor.predict(X_test)

X_train_flat = X_train_extracted.reshape(X_train_extracted.shape[0], -1)
X_test_flat = X_test_extracted.reshape(X_test_extracted.shape[0], -1)

y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train_flat, y_train_labels)

rf_predictions = clf.predict(X_test_flat)
rf_accuracy = accuracy_score(y_test_labels, rf_predictions)

print(f"Original Model Accuracy: {cnn_accuracy[1]:4f}")
print(f"Random Forest Accuracy: {rf_accuracy:4f}")
