import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys
import os
import csv
from utils import (plot_feature_corr, plot_clusters,
                   plot_pca, plot_feature_importance,
                   pr_curve, print_classfication_report,
                   read_data, write_data)

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.decomposition import KernelPCA
from utils import score_model
#------------------------------------------------------------------------------
# Reading Data
# Please see file utils.py for read_data method

X_train, y_train, X_test = read_data()



# Standardezing data------------------------------------------------------------

# Standerizing data by making it zero mean and unit variance
# dividing the training data in two section to avoid overfitting
# Grid Search with crossvalidation will run on training data and will be evaluated on dev set


skf = StratifiedKFold(y_train, n_folds=4)
train_index, dev_index = next(iter(skf))


X_dev = X_train[dev_index]
y_dev = y_train[dev_index]

X_train = X_train[train_index]
y_train = y_train[train_index]



scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = np.array([scaler.transform(block) for block in X_test])
X_dev = scaler.transform(X_dev)


# X = np.vstack((X_train, X_test))
# X = StandardScaler().fit_transform(X)
# X_train = X[:X_train.shape[0]]
# X_test = X[X_train.shape[0]:]

X_train, y_train = shuffle(X_train, y_train)


#------------------------------------------------------------------------------
# Features Engineering Diagnostic plots

plot_feature_corr(X_train)
plot_feature_corr(np.vstack(X_test), stem='test')
plot_pca(X_train, y_train)
plot_clusters(X_train, y_train)
indices_ci = plot_feature_importance(X_train, y_train)

# n_f = 10
#
#
# X_train = X_train[:, indices_ci[:n_f]] # Six most important Features
# X_dev = X_dev[:, indices_ci[:n_f]]
# X_test = X_test[:, indices_ci[:n_f]]

# kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
# kp = kpca.fit(X_train)
# X_train = kp.transform(X_train)
# X_test = kp.transform(X_test)
# X_dev = kp.transform(X_dev)


# Will not select features at this stage,
# Classifiers will them self select important features through regularization

n_components = len(np.unique(y_train))

#------------------------------------------------------------------------------
# Classifiers

# class_ws = [1, 5, 1, 1, 1, 1, 1, 5, 1]
# class_weight = {float(index):ws for index,ws in enumerate(class_ws)}

clf1 = KNeighborsClassifier()
clf1a = KNeighborsClassifier(metric_params={'V': np.cov(X_train.T)}, algorithm='brute')
clf2 = RandomForestClassifier()
clf3 = ExtraTreesClassifier(class_weight='balanced')
clf4 = MLPClassifier()
clf5 = GaussianMixture(n_components=n_components,
                                           n_init=10, max_iter=500)
clf6 = SVC(probability=True)
clf7 = AdaBoostClassifier(algorithm='SAMME')
clf8 = LogisticRegression(multi_class='multinomial', max_iter=500, penalty = 'l2', class_weight='balanced')



# Cross-validation Scheme
cv = StratifiedKFold(y_train, n_folds=3, shuffle=True)


# KN Grid Search---------------------------------------------------------------
metrics       = ['euclidean','manhattan']
weights       = ['uniform','distance']
numNeighbors  = [12,15,17]
param_grid    = dict(metric=metrics, weights=weights, n_neighbors=numNeighbors)

CV_kn = GridSearchCV(clf1, param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_kn = CV_kn.fit(X_train, y_train)
print(CV_kn.best_estimator_)
kn = CV_kn.best_estimator_

y_pred_train_kn = cross_val_predict(kn, X_train, y_train, cv=cv)
print_classfication_report(kn, y_train, y_pred_train_kn, stem='kn')
pr_curve(y_train, y_pred_train_kn, n_components, 'kn')

y_pred_dev_kn = kn.predict(X_dev)
print_classfication_report(kn, y_dev, y_pred_dev_kn, stem='kn_dev')
pr_curve(y_dev, y_pred_dev_kn, n_components, 'kn_dev')

# KN Grid Search (Backward compatability bug in scikit)----------------------------------------------
metrics       = ['mahalanobis']
weights       = ['uniform','distance']
numNeighbors  = [13,15,17]
param_grid    = dict(metric=metrics, weights=weights, n_neighbors=numNeighbors)

CV_km = GridSearchCV(clf1a, param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_km = CV_km.fit(X_train, y_train)
print(CV_km.best_estimator_)
km = CV_km.best_estimator_

y_pred_train_km = cross_val_predict(km, X_train, y_train, cv=cv)
print_classfication_report(km, y_train, y_pred_train_km, stem='km')
pr_curve(y_train, y_pred_train_km, n_components, 'km')

y_pred_dev_km = km.predict(X_dev)
print_classfication_report(km, y_dev, y_pred_dev_km, stem='km_dev')
pr_curve(y_dev, y_pred_dev_km, n_components, 'km_dev')

# Random Forest----------------------------------------------------------------
n_estimators = [200, 500, 1000]
max_features = ['sqrt', 5, 9]
param_grid    = dict(n_estimators=n_estimators, max_features=max_features)

CV_rf = GridSearchCV(clf2, param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_rf = CV_rf.fit(X_train, y_train)
print(CV_rf.best_estimator_)
rf = CV_rf.best_estimator_

y_pred_train_rf = cross_val_predict(rf, X_train, y_train, cv=cv)
print_classfication_report(rf, y_train, y_pred_train_rf, stem='rf')
pr_curve(y_train, y_pred_train_rf, n_components, 'rf')



y_pred_dev_rf = rf.predict(X_dev)
print_classfication_report(rf, y_dev, y_pred_dev_rf, stem='rf_dev')
pr_curve(y_dev, y_pred_dev_rf, n_components, 'rf_dev')



# Visualizing partially first two trees
forest = rf.estimators_[:2]
for index, tree in enumerate(forest):
    export_graphviz(tree,
                feature_names=map(str, range(X_train.shape[1])),
                max_depth=3,
                filled=True,
                rounded=True,
                out_file=str(index) + '_tree.dot')
    os.system('dot -Tpng '  + str(index) + '_tree.dot -o ' + str(index) + '_tree.png')


# Extra Trees-----------------------------------------------------------------
n_estimators = [300, 400, 500]
criterion = ['gini']
max_depth = range(1, 18, 3)
min_samples_split = [2, 4, 8, 10]
max_features = ['sqrt', 5, 9, 11]
param_grid    = dict(criterion=criterion, max_depth=max_depth,
                     min_samples_split=min_samples_split,
                     n_estimators=n_estimators, max_features=max_features)

CV_et = GridSearchCV(clf3, param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_et = CV_et.fit(X_train, y_train)
print(CV_et.best_estimator_)
et = CV_et.best_estimator_

y_pred_train_et = cross_val_predict(et, X_train, y_train, cv=cv)
print_classfication_report(et, y_train, y_pred_train_et, stem='et')
pr_curve(y_train, y_pred_train_et, n_components, 'et')


y_pred_dev_et = et.predict(X_dev)
print_classfication_report(et, y_dev, y_pred_dev_et, stem='et_dev')
pr_curve(y_dev, y_pred_dev_et, n_components, 'et_dev')
# MLP-----------------------------------------------------------------------
alpha = 10.0 ** -np.arange(1, 3)
hidden_layer_sizes = [(120,), (115,), (125)]
solver = ['lbfgs', 'adam']
activation = ['logistic', 'tanh', 'relu']
param_grid    = dict(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, solver=solver, activation=activation)
CV_ml = GridSearchCV(clf4, param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_ml = CV_ml.fit(X_train, y_train)
print(CV_ml.best_estimator_)
ml = CV_ml.best_estimator_

y_pred_train_ml = cross_val_predict(ml, X_train, y_train, cv=cv)
print_classfication_report(ml, y_train, y_pred_train_ml, stem='ml')
pr_curve(y_train, y_pred_train_ml, n_components, 'ml')


y_pred_dev_ml = ml.predict(X_dev)
print_classfication_report(ml, y_dev, y_pred_dev_ml, stem='ml_dev')
pr_curve(y_dev, y_pred_dev_ml, n_components, 'ml_dev')
# GMM -----------------------------------------------------------------------
covariance_type = ['tied','full','spherical', 'diag']
param_grid    = {'estimator__covariance_type':covariance_type, 'estimator__n_components':[n_components]}
#CV_gm = GridSearchCV(OneVsRestClassifier(clf5), param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_gm = GridSearchCV(OneVsOneClassifier(clf5), param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_gm = CV_gm.fit(X_train, y_train)
print(CV_gm.best_estimator_)
gm = CV_gm.best_estimator_

y_pred_train_gm = cross_val_predict(gm, X_train, y_train, cv=cv)
print_classfication_report(gm, y_train, y_pred_train_gm, stem='gm')
pr_curve(y_train, y_pred_train_gm, n_components, 'gm')

y_pred_dev_gm = gm.predict(X_dev)
print_classfication_report(gm, y_dev, y_pred_dev_gm, stem='gm_dev')
pr_curve(y_dev, y_pred_dev_gm, n_components, 'gm_dev')
# SVC -----------------------------------------------------------------------
gamma_range = [0.01, 0.15, 0.2]
#C_range = [1, 10, 30, 50]
C_range = [1, 2.5, 3.5]
kernel = ['rbf', 'linear']
dec_func = ['ovr']
param_grid    =  {'estimator__C':C_range, 'estimator__kernel':kernel, 'estimator__probability':[True],
                  'estimator__gamma':gamma_range}
#CV_sv = GridSearchCV(OneVsRestClassifier(clf6), param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_sv = GridSearchCV(OneVsOneClassifier(clf6), param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_sv = CV_sv.fit(X_train, y_train)
print(CV_sv.best_estimator_)
sv = CV_sv.best_estimator_

y_pred_train_sv = cross_val_predict(sv, X_train, y_train, cv=cv)
print_classfication_report(sv, y_train, y_pred_train_sv, stem='sv')
pr_curve(y_train, y_pred_train_sv, n_components, 'sv')


y_pred_dev_sv = sv.predict(X_dev)
print_classfication_report(sv, y_dev, y_pred_dev_sv, stem='sv_dev')
pr_curve(y_dev, y_pred_dev_sv, n_components, 'sv_dev')


# OneVsRestClassifier N (n_classes) classifiers, one vs one (N*(N-1))/2 classifiers
# Adaboost -----------------------------------------------------------------------
n_estimators = range(100, 501, 100)
learning_rate = np.logspace(-4, 0, 4)
max_depth_list = range(1, 11, 3)
base_classifier_list = [DecisionTreeClassifier(max_depth=x, max_features='auto') for x in max_depth_list]

param_grid    =  dict(n_estimators=n_estimators, base_estimator=base_classifier_list,
                  learning_rate=learning_rate)
CV_ab = GridSearchCV(clf7, param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_ab = CV_ab.fit(X_train, y_train)
print(CV_ab.best_estimator_)
ab = CV_ab.best_estimator_

y_pred_train_ab = cross_val_predict(ab, X_train, y_train, cv=cv)
print_classfication_report(ab, y_train, y_pred_train_ab, stem='ab')
pr_curve(y_train, y_pred_train_ab, n_components, 'ab')

y_pred_dev_ab = ab.predict(X_dev)
print_classfication_report(ab, y_dev, y_pred_dev_ab, stem='ab_dev')
pr_curve(y_dev, y_pred_dev_ab, n_components, 'ab_dev')
# Logistic Regression -----------------------------------------------------------------------

solver = ['newton-cg', 'lbfgs', 'sag']
C = [0.1, 1, 2, 3, 4, 5]

param_grid    =  dict(solver=solver,
                  C=C)
CV_lr = GridSearchCV(clf8, param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_lr = CV_lr.fit(X_train, y_train)
print(CV_lr.best_estimator_)
lr = CV_lr.best_estimator_

y_pred_train_lr = cross_val_predict(lr, X_train, y_train, cv=cv)
print_classfication_report(lr, y_train, y_pred_train_lr, stem='lr')
pr_curve(y_train, y_pred_train_lr, n_components, 'lr')

y_pred_dev_lr = lr.predict(X_dev)
print_classfication_report(lr, y_dev, y_pred_dev_lr, stem='lr_dev')
pr_curve(y_dev, y_pred_dev_lr, n_components, 'lr_dev')
# Soft Voting-----------------------------------------------------------------
param_grid    =  {'estimator__C':C_range, 'estimator__kernel':kernel, 'estimator__probability':[True],
                  'estimator__gamma':gamma_range}
CV_sv = GridSearchCV(OneVsRestClassifier(clf6), param_grid=param_grid, cv=cv, n_jobs=3, verbose=3, scoring='f1_weighted')
CV_sv = CV_sv.fit(X_train, y_train)
sv1 = CV_sv.best_estimator_
es = VotingClassifier(estimators=[('kn', kn), ('km', km), ('rf', rf),('lr', lr),
                                    ('et', et), ('ml', ml), ('sv', sv1), ('ab', ab)],
                      weights=[1, 1, 1, 1.1, 1.1, 4, 4, 1.1], voting='soft')
es = es.fit(X_train, y_train)
y_pred_train_es = cross_val_predict(es, X_train, y_train, cv=cv)
print_classfication_report(es, y_train, y_pred_train_es, stem='es')
pr_curve(y_train, y_pred_train_es, n_components, 'es')


y_pred_dev_es = es.predict(X_dev)
print_classfication_report(es, y_dev, y_pred_dev_es, stem='es_dev')
pr_curve(y_dev, y_pred_dev_es, n_components, 'es_dev')


# Hard Voting-----------------------------------------------------------------
eh = VotingClassifier(estimators=[('kn', kn), ('km', km), ('rf', rf),('lr', lr),
                                    ('et', et), ('ml', ml), ('sv', sv), ('ab', ab)],
                      weights=[1, 1, 1, 1.1, 1.1, 4, 4, 1.1], voting='hard')
eh = eh.fit(X_train, y_train)

y_pred_train_eh = cross_val_predict(eh, X_train, y_train, cv=cv)
print_classfication_report(eh, y_train, y_pred_train_eh, stem='eh')
pr_curve(y_train, y_pred_train_eh, n_components, 'eh')


y_pred_dev_eh = eh.predict(X_dev)
print_classfication_report(eh, y_dev, y_pred_dev_eh, stem='eh_dev')
pr_curve(y_dev, y_pred_dev_eh, n_components, 'eh_dev')

# Generating predictions-----------------------------------------------------------------

labels_test = es.predict(X_test)

blocks_pred, blocks_consensus = score_model(es, X_test)


#------------------------------------------------------------------------------
# Saving Prediction, may the force is with me


write_data(blocks_pred, 'khan_speaker_labels.csv')

# np.savetxt('khan_speaker_labels.csv', labels_test,
#            delimiter=' ', fmt='%i', header= 'block_num, prediction', comments='')
#
#
# with open('khan_speaker_labels_alternate.csv', 'wb') as fh:
#     writer = csv.writer(fh, delimiter=',')
#     writer.writerow(['block_num', 'prediction'])
#     writer.writerows(labels_test)









