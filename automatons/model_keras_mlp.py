from __future__ import print_function
import keras
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.cross_validation import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam, Nadam # Just trying different optimizer
from utils import (plot_feature_corr, plot_clusters,
                   plot_pca, plot_feature_importance,
                   pr_curve, print_classfication_report,
                   read_data, write_data, score_model)
from sklearn import metrics


batch_size = 128
epochs = 50

X_train, y_train, X_test = read_data()



# Feature Diagnostic

plot_feature_corr(X_train)
plot_feature_corr(np.vstack(X_test), stem='test')
plot_pca(X_train, y_train)
plot_clusters(X_train, y_train)
indices_ci = plot_feature_importance(X_train, y_train)

skf = StratifiedKFold(y_train, n_folds=4)
train_index, dev_index = next(iter(skf))


X_dev = X_train[dev_index]
y_dev = y_train[dev_index]

X_train = X_train[train_index]
y_train = y_train[train_index]

# Since GMM works well, transforming the data to alternate space
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
kp = kpca.fit(X_train)
X_train = kp.transform(X_train)
X_dev = kp.transform(X_dev)
X_test = np.array([kp.transform(block) for block in X_test])


num_classes = len(np.unique(y_train))
num_features = X_train.shape[1]

# convert class vectors to binary class matrices
# keras one hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_dev = keras.utils.to_categorical(y_dev, num_classes)



# tried tanh, and relu, tanh was better on development set
model = Sequential()
model.add(Dense(128, activation='tanh', input_shape=(num_features,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
yp = model.predict(X_dev)

yp= np.argmax(yp, axis=1)
y_dev = np.argmax(y_dev, axis=1)



#performance on dev set
print_classfication_report('Keras', y_dev, yp, stem='keras_dev')
pr_curve(y_dev, yp, num_classes, 'keras_dev')



# prediction is done for every sample and
# prediction for whole block is done through consensus voting
# blocks_consensus tell me about block prediction distrubution
# making sure there is no bimodal distribution
blocks_pred, blocks_consensus = score_model(model, X_test)

#writing csv file
write_data(blocks_pred, 'khan_speaker_labels_MLP.csv')