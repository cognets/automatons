import numpy as np
import sys
from sklearn import mixture
from utils import write_data, read_data_blocks
from utils import pr_curve, print_classfication_report, plot_feature_importance
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_model(features):
    mixture_count = features.shape[1] - 2  
    gmm = mixture.GaussianMixture(n_components=mixture_count, covariance_type='diag', n_init=1)
    gmm.fit(features)
    return gmm


def run_model(X, y):
    speakers = []
    n_speakers = len(np.unique(y))
    for speaker_id in range(n_speakers):
            gmm = build_model(np.vstack(X[y==speaker_id]))
            speaker = {'name':speaker_id, 'gmm':gmm}
            speakers.append(speaker)
    return speakers

def score_models(speakers, utterances):
    small_number = -sys.float_info.max
    yp = np.empty((utterances.shape[0],))
    for index, sample in enumerate(utterances):
        best_score = small_number #the smallest (i.e., most neghative) float number
        for speaker in speakers: #find the most similar known speaker for the given test sample of a voice
            score = speaker['gmm'].score(sample) #yields log-likelihoods per feature vector
            #score = np.sum(score_per_featurevector) #...these can be aggregated to a score per feature-set by summing
            if score > best_score:
                best_score = score
                yp[index] = np.int32(speaker['name'])

    return yp

# Standardezing data------------------------------------------------------------

train_blocks, train_blocks_labels, train_labels, test_blocks = read_data_blocks()

X_train = np.vstack(train_blocks)
n_classes = len(np.unique(train_labels))

ind = 12
indices_ci = plot_feature_importance(X_train, train_labels)


# scaler = StandardScaler().fit(X_train[:,indices_ci[:ind]])
#
# train_blocks = np.array([scaler.transform(block[:,indices_ci[:ind]]) for block in train_blocks])
# test_blocks  = np.array([scaler.transform(block[:,indices_ci[:ind]]) for block in test_blocks])
# Decided to use all features as each of them carry some unique information

scaler = StandardScaler().fit(X_train)

train_blocks = np.array([scaler.transform(block) for block in train_blocks])
test_blocks  = np.array([scaler.transform(block) for block in test_blocks])


X_train_blocks, X_dev_blocks, y_train_blocks, y_dev_blocks = train_test_split(train_blocks, train_blocks_labels, test_size=0.2, random_state=455)

speakers = run_model(X_train_blocks, y_train_blocks)



yp = score_models(speakers, X_dev_blocks)

print_classfication_report('gmm', y_dev_blocks, yp, stem='gmm_dev')
pr_curve(y_dev_blocks, yp, n_classes, 'gmm_dev')
plt.close('all')


blocks_pred = score_models(speakers, test_blocks)
write_data(blocks_pred, 'khan_speaker_labels_GMM.csv')

