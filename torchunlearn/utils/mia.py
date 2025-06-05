import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch

def get_stratified_set(x, y, data_size, seed):
    sss_retain = StratifiedShuffleSplit(n_splits=1, test_size=data_size, random_state=seed)

    for retain_train_idx, retain_test_idx in sss_retain.split(x, y):
        stratified_x_retain = x[retain_test_idx]
        stratified_y_retain = y[retain_test_idx]

    return stratified_x_retain, stratified_y_retain

def create_stratified_balanced_set(logits, seed=42, data_size=None, prob_only=False):
    np.random.seed(seed)
    
    x_retain, y_retain = logits['(R)']
    x_test, y_test = logits['(Te)']
    x_forget, y_forget = logits['(F)']

    # in case of using forget test dataset
    # x_test2, y_test2 = logits['(TF)']
    # x_test = torch.cat([x_test, x_test2])
    # y_test = torch.cat([y_test, y_test2])
        
    num_retain = len(x_retain)
    num_test = len(x_test)

    if data_size is None:
        data_size = min(num_retain, num_test)
    
    if data_size != num_retain:
        x_retain, y_retain = get_stratified_set(x_retain, y_retain, data_size, seed)        
    
    if data_size != num_test:
        x_test, y_test = get_stratified_set(x_test, y_test, data_size, seed) 

    x_train = torch.cat([x_retain, x_test])
    y_train = torch.cat([y_retain, y_test])

    if prob_only:
        softmax = torch.nn.Softmax(dim=1)
        x_train = torch.gather(softmax(x_train), 1, y_train[:, None])
        x_forget = torch.gather(softmax(x_forget), 1, y_forget[:, None])

    y_train[:len(y_retain)] = 1  # used during training
    y_train[len(y_retain):] = 0  # not-used during training
    y_forget = y_forget*0  # not-used during training
    
    return x_train, y_train, x_forget, y_forget

from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
def MIA_SVC(logits, oneclass=False, seed=42, data_size=None, prob_only=True):
    x_train, y_train, x_forget, y_forget = create_stratified_balanced_set(logits, seed, data_size, prob_only)

    if oneclass:
        clf = OneClassSVM(gamma='auto')
        clf.fit(x_train)
    else:
        clf = SVC(C=3, gamma="auto", kernel="rbf")
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_train)
    if isinstance(clf, OneClassSVM):
        y_pred = (y_pred + 1)//2  # since OneClassSVM returns -1, 1

    cm = confusion_matrix(y_train, y_pred, labels=[0, 1])
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    res = {}
    
    train_acc = (TP + TN) / (TP + TN + FP + FN)
    # print("Train Acc", train_acc)

    y_pred = clf.predict(x_forget)
    if isinstance(clf, OneClassSVM):
        y_pred = (y_pred + 1)//2  # since OneClassSVM returns -1, 1
        
    cm = confusion_matrix(y_forget, y_pred, labels=[0, 1])
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    test_acc = (TP + TN) / (TP + TN + FP + FN)
    # print("Test Acc", test_acc)
    mia_eff = TN/len(x_forget)
    # print("MIA Efficiency", mia_eff)

    res["MIA Train Acc"] = train_acc
    res["MIA Test Acc"] = test_acc
    res["MIA Efficiency"] = mia_eff
    
    return res