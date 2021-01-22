## -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import sys
import socket
import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.utils import shuffle

import encdec

def recv(soc, buffer_size = 4096, recv_timeout = 30):
    received_data = b""
    while True:
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
            if received_data.decode('utf-8')[-1] == '\4':
                break
        except socket.timeout:
            return None, 0
        except BaseException as e:
            return None, 0
    try:
        received_data = received_data.decode('utf-8')[:-1]
        received_data = json.loads(received_data, cls = encdec.MessageDecoder)
    except BaseException as e:
        return None, 0
    return received_data, 1



bim = pd.read_csv('./MergeM3.bim', header = None, sep = '\t') # CHANGE WITH THE NAME OF THE MASK TO USE
ped = pd.read_csv('./MergeM3.ped', header = None, index_col = 0, sep = '\t') # CHANGE WITH THE NAME OF THE MASK TO USE
pheno = pd.read_csv('./pheno.txt', sep = ' ', index_col = 0) # CHANGE WITH THE NAME OF THE PHENOTYPE FILE, FOR AN EXAMPLE SEE example_pheno.txt
covar_unnorm = pd.read_csv('./covar.txt', sep = ' ', index_col = 0) # CHANGE WITH THE NAME OF THE COVARIATES FILE, FOR AN EXAMPLE SEE example_covar.txt
server_ip = 'localhost' # CHANGE WITH THE IP ADDRESS OF THE SERVER, localhost for testing the code locally
contact_id = 'blablabla' # CHANGE TO YOUR EMAIL, OR ANYTHING THAT ALLOWS ME TO CONTACT YOU IN CASE THE CODE DOESN'T WORK

server_port = 10000
ped.drop(ped.columns[range(5)], axis = 1, inplace = True)
ref_ref = (bim[5] + ' ' + bim[5]).values.reshape((1,-1)) 
feature_genes = pd.DataFrame((ped.values != ref_ref).astype(int), columns = bim[1], index = ped.index)
age_max = 100
index_samples = set(feature_genes.index) & set(pheno.index) & set(covar_unnorm.index)
phenotypes = pheno.loc[index_samples]['B2']
covar = covar_unnorm.loc[index_samples][['age','sex']]
covar['age'][covar['age'] > age_max] = age_max
covar['age'] = covar['age'] / age_max
covar['age*sex'] = covar['age']*covar['sex']
covar['age^2'] = covar['age']**2
covar['age^2*sex'] = covar['age']**2*covar['sex']
X = np.concatenate((covar.values, feature_genes.loc[index_samples].values), axis = 1)
y = phenotypes.values
feature_names = np.array(['age','sex','age*sex','age^2','age^2*sex'] + list(feature_genes.columns))
print('Number of samples: {} {}'.format(X.shape[0], len(y)))
print('Number of features: {} {}'.format(X.shape[1], len(feature_names)))
print('Samples per phenotype class: {}'.format(', '.join(['{} = {}'.format(class_id, n_samples) for class_id, n_samples in zip(*np.unique(y, return_counts = True))])))

while True:
    soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    try:
        soc.connect((server_ip, server_port))
    except BaseException as e:
        time.sleep(10)
        soc.close()
    while True:
        received_message, status = recv(soc=soc)
        if status == 0:
            break
        response_message = {'version':received_message['version']}
        if received_message['what2do'] == 'split':
            skf = StratifiedKFold(n_splits = received_message['k_folds'])
            train_indexes, test_indexes = [], []
            for train_index, test_index in skf.split(X, y):
                train_indexes.append(train_index)
                test_indexes.append(test_index)
            response_message['what2do'] = 'ready'
        elif received_message['what2do'] == 'train_all_samples':
            train_indexes, test_indexes = [range(X.shape[0]),], [range(X.shape[0]),] # these are not really used as training/testing set, it's just for consistency with the 'minimize' section in cross-validation steps
            response_message['what2do'] = 'ready'
        elif received_message['what2do'] == 'minimize':
            model = received_message['model']
            if isinstance(received_message['coef_'], pd.Series):
                model.coef_ = received_message['coef_'].reindex(feature_names, fill_value = 0.0)
                model.intercept_ = np.array([model.intercept_,])
            i_fold = received_message['i_fold']
            if i_fold >= len(train_indexes):
                response_message['what2do'] == 'echo'
            else:
                train_index = train_indexes[i_fold]
                test_index = test_indexes[i_fold]
                for j in range(1):
                    model.partial_fit(X[train_index,:], y[train_index], classes = [0,1])
                cnf = confusion_matrix(y[test_index], model.predict(X[test_index,:]))
                acc = np.sum(np.diag(cnf)) / np.sum(cnf)
                loss = log_loss(y[train_index], model.predict_proba(X[train_index,:]), eps=1e-15)
                loss_reg = loss + model.get_params()['alpha']*np.sum(np.abs(model.coef_))
                #print('DEBUG> {}\t{}\t{}\t{}\t{}\t{}'.format(received_message['version'], received_message['i_iter'], i_fold, loss, loss_reg, acc))
                response_message['what2do'] = 'update'
                response_message['n_samples'] = len(train_index)
                response_message['confusion_matrix'] = cnf 
                response_message['loss'] = loss
                response_message['loss_reg'] = loss_reg
                response_message['model'] = model
                response_message['coef_'] = pd.Series(model.coef_.flatten(), index = feature_names)
        elif received_message['what2do'] == 'echo':
            response_message = received_message
        elif received_message['what2do'] == 'reset':
            response_message = received_message
            response_message['contact_id'] = contact_id
        elif received_message['what2do'] == 'done':
            break
        else:
            break
        response_json = json.dumps(response_message, cls = encdec.MessageEncoder) + '\4'
        response_bin = bytes(response_json, encoding="utf-8")
        try:
            soc.sendall(response_bin)
        except:
            break
    soc.close()
