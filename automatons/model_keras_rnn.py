from __future__ import print_function
import numpy as np

from keras.optimizers import SGD

import keras
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from utils import read_data
from sklearn.decomposition import KernelPCA
from sklearn.cross_validation import StratifiedKFold


hidden_units = 10

epochs = 5

X_train, y_train, X_test_report = read_data()


skf = StratifiedKFold(y_train, n_folds=3)
train_index, dev_index = next(iter(skf))


X_dev = X_train[dev_index]
y_dev = y_train[dev_index]

X_train = X_train[train_index]
y_train = y_train[train_index]


kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
kp = kpca.fit(X_train)
X_train = kp.transform(X_train)
X_dev = kp.transform(X_dev)




xts = X_train.shape
X_train = np.reshape(X_train, (xts[0], xts[1], 1))
xtss = X_dev.shape
X_dev = np.reshape(X_dev, (xtss[0], xtss[1], 1))

y_train =y_train[:, np.newaxis]
y_dev = y_dev[:, np.newaxis]


batch_size = X_train.shape[0]
num_classes = len(np.unique(y_train))
num_features = X_train.shape[1]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_dev = keras.utils.to_categorical(y_dev, num_classes)


model = Sequential()

#batch_input_shape= (batch_size, X_train.shape[1], X_train.shape[2])

# note that it is necessary to pass in 3d batch_input_shape if stateful=True
model.add(LSTM(64, return_sequences=True, stateful=False,
               batch_input_shape= (X_train.shape[0], X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences=True, stateful=False))
model.add(LSTM(64, stateful=False))


# add dropout to control for overfitting
model.add(Dropout(.5))

# squash output onto number of classes in probability space
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_dev, y_dev))

y_pred=model.predict_classes(X_dev, batch_size=batch_size)
print(classification_report(y_dev, y_pred))