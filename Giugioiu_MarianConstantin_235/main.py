import pdb
import numpy as np
import math
import sklearn
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt 
from scipy.io.wavfile import read
import librosa as librosa
import librosa.display
import IPython.display as ipd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, classification_report


#citirea si generarea caracteristicilor si etichetelor pentru train
train_files=[]
train_labels=[]
file = open("train.txt")
for line in file:
    l=(line.split(","))
    train_files.append(l[0])
    train_labels.append(float(l[1][:1]))

n1=len(train_files)
y_train = np.array(train_labels[:n1])

train_features = []
for nume in train_files[:n1]:
    x, sr = librosa.load('train/train/'+nume,sr=None)
    mfcc = librosa.feature.mfcc(x, sr=sr)
    train_features.append(mfcc.flatten())

x_train=np.array(train_features[:n1])

#citirea si generarea caracteristicilor si etichetelor pentru validare
validation_files=[]
validation_labels=[]
file = open("validation.txt")
for line in file:
    l=(line.split(","))
    validation_files.append(l[0])
    validation_labels.append(float(l[1][:1]))

n2=len(validation_files)
y_validation = np.array(validation_labels[:n2])

validation_features = []
for nume in validation_files[:n2]:
    x, sr = librosa.load('validation/validation/'+nume,sr=None)
    mfcc = librosa.feature.mfcc(x, sr=sr)  
    validation_features.append(mfcc.flatten())

x_validation=np.array(validation_features[:n2])

#citirea si generarea caracteristicilor si etichetelor pentru test
test_files=[]
file = open("test.txt")
for line in file:
    test_files.append(line[:len(line)-1])

n3=len(test_files)

test_features = []
for nume in test_files[:n3]:
    x, sr = librosa.load('test/test/'+nume,sr=None)
    mfcc = librosa.feature.mfcc(x, sr=sr)  
    test_features.append(mfcc.flatten())

x_test=np.array(test_features[:n3])

#calcularea acuratetii
def compute_accuracy(gt_labels, predicted_labels):
    accuracy = np.sum(predicted_labels == gt_labels) / len(predicted_labels)
    return accuracy

#metodele de normalizare a datelor
def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data) 
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data)

#normalizarea seturilor de date training, test si validation
scaled_train_data, scaled_test_data = normalize_data(x_train, x_test, type='standard')
scaled_train_data, scaled_validation_data = normalize_data(x_train, x_validation, type='standard')


#model Naive Bayes
nb_model = GaussianNB()
#antrenare
nb_model.fit(scaled_train_data, y_train)
predicted_labels_nb=nb_model.predict(scaled_validation_data)

#acuratete
model_accuracy_nb = compute_accuracy(y_validation, predicted_labels_nb)
print('NP model accuracy: ', model_accuracy_nb)

#matricea de confuzie
conf_matrix = np.zeros((2, 2))
for i in range(len(y_validation)): 
    conf_matrix[int(y_validation[i]), int(predicted_labels_nb[i])] +=1
print(conf_matrix)

#precizie, recall
print(classification_report(y_validation, predicted_labels_nb))


#model masini cu vector suport
svm_model = svm.NuSVC(nu=0.1,kernel = 'rbf',gamma=0.001)

#antrenare
svm_model.fit(scaled_train_data, y_train)
predicted_labels_svm = svm_model.predict(scaled_validation_data)

#acuratete
model_accuracy_svm = compute_accuracy(y_validation, predicted_labels_svm)
print("SVM model accuracy: ", model_accuracy_svm)

#matricea de confuzie
conf_matrix = np.zeros((2, 2))
for i in range(len(y_validation)): 
    conf_matrix[int(y_validation[i]), int(predicted_labels_svm[i])] +=1
print(conf_matrix)

#precizie, recall
print(classification_report(y_validation, predicted_labels_svm))


#scrierea rezultatului in fisierul csv
predicted_labels = svm_model.predict(scaled_test_data)
with open('submission.csv', mode='w') as csv_file:
    fieldnames = ['name','label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(n3):
        writer.writerow({'name': test_files[i], 'label': int(predicted_labels[i])})









