# -*- coding: utf-8 -*-

# This code is an adaption of: https://github.com/ahmedfgad/FederatedLearning.git

import socket
import json
import time
import threading
import numpy as np
import pandas as pd
import functools
print = functools.partial(print, flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import encdec
thread_lock = threading.Lock()

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size = 4096, recv_timeout = 5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data
                if data == b'':
                    received_data = b""
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0
                elif data.decode('utf-8')[-1] == '\4': # end of transmission
                    #print('All data ({data_len} bytes) Received from {client_info}.'.format(client_info=self.client_info, data_len=len(received_data)))
                    if len(received_data) > 0:
                        try:
                            received_data = received_data.decode('utf-8')[:-1]
                            received_data = json.loads(received_data, cls = encdec.MessageDecoder)
                            return received_data, 1
                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    self.recv_start_time = time.time()
            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def reply(self, received_message):
        global server_model, server_coef_, client_infos, confusion_matrix, acc_means, acc_stds, i_fold, k_folds, i_iter, i_reg, i_reg_best,  max_iter, regs, server_model_version
        thread_lock.acquire()
        if (type(received_message) is dict):
            response_message = {'what2do':'done', 'version':server_model_version, 'i_iter': i_iter}
            if (i_iter == max_iter):
                for client_name, client_score in client_infos['scores'].items():
                    confusion_matrix[:,:,i_fold] += client_score
                acc = np.sum(np.diag(confusion_matrix[:,:,i_fold])) / np.sum(confusion_matrix[:,:,i_fold])
                print('fold {} Done --> accuracy = {}'.format(i_fold, acc))
                client_infos['n_samples'] = {}
                client_infos['scores']= {}
                client_infos['coef_'] = {}
                client_infos['models'] = {}
                i_iter = 0
                i_fold += 1
                response_message['i_iter'] = i_iter
                if i_fold == k_folds:
                    i_fold = 0
                    acc = np.array([np.sum(np.diag(confusion_matrix[:,:,i])) / np.sum(confusion_matrix[:,:,i]) for i in range(k_folds)])
                    acc_means[i_reg] = np.mean(acc)
                    acc_stds[i_reg] = np.std(acc)
                    print('reg. strength = {} --> accuracy = {} +/- {}'.format(regs[i_reg], acc_means[i_reg], acc_stds[i_reg]))
                    confusion_matrix = np.zeros((2,2,k_folds))
                    i_reg += 1
                    if i_reg == len(regs):
                        i_reg_best = np.argmax(acc_means)
                        print('Cross-validation results:')
                        print('\treguralization strengths:', regs)
                        print('\taccuracy means:', acc_means)
                        print('\taccuracy stds:', acc_stds)
                        print('\toptimal regularization:', regs[i_reg_best])
                        server_model = SGDClassifier(loss = 'log', penalty = 'l1', alpha = regs[i_reg_best], max_iter = 1, learning_rate = 'constant', eta0 = learning_rate_0, early_stopping = False, tol = None)
                        server_coef_ = False
                        response_message['what2do'] = 'train_all_samples'
                    else:
                        server_model = SGDClassifier(loss = 'log', penalty = 'l1', alpha = regs[i_reg], max_iter = 1, learning_rate = 'constant', eta0 = learning_rate_0, early_stopping = False, tol = None)
                        server_coef_ = False
                        response_message['what2do'] = 'minimize'
                        response_message['model'] = server_model
                        response_message['coef_'] = server_coef_
                        response_message['i_fold'] = i_fold
                else:
                    if i_reg == len(regs):
                        feature_names = server_coef_.index
                        coef_values = server_coef_.values
                        ind_top = np.argsort(np.abs(coef_values))[::-1][:10]
                        response_message['what2do'] = 'done'
                        print('ESTIMATED ACCURACY: {} ± {}'.format(acc_means[i_reg_best], acc_stds[i_reg_best]))
                        print('TOP FEATURES:',feature_names[ind_top])
                        print('TOP COEFFICIENTS:',coef_values[ind_top])
                        print('NUMBER OF FEATURES:',len(feature_names))
                        server_coef_.to_csv('coef.csv')
                    else:
                        server_model = SGDClassifier(loss = 'log', penalty = 'l1', alpha = regs[i_reg], max_iter = 1, learning_rate = 'constant', eta0 = learning_rate_0, early_stopping = False, tol = None)
                        server_coef_ = False
                        response_message['what2do'] = 'minimize'
                        response_message['model'] = server_model
                        response_message['coef_'] = server_coef_
                        response_message['i_fold'] = i_fold
            else:
                if received_message['what2do'] == 'ready':
                    response_message['what2do'] = 'minimize'
                    response_message['model'] = server_model
                    response_message['coef_'] = server_coef_
                    response_message['i_fold'] = i_fold
                elif received_message['what2do'] == 'update':
                    if received_message['version'] == response_message['version']:
                        client_infos['models'][self.client_info] = received_message['model']
                        client_infos['coef_'][self.client_info] = received_message['coef_']
                        client_infos['scores'][self.client_info] = received_message['confusion_matrix']
                        client_infos['n_samples'][self.client_info] = received_message['n_samples']
                        update_server_model(self.client_info)
                        response_message['what2do'] = 'echo'
                    else:
                        response_message['what2do'] = 'minimize'
                        response_message['model'] = server_model
                        response_message['coef_'] = server_coef_
                        response_message['i_fold'] = i_fold
                elif received_message['what2do'] == 'reset':
                        response_message['what2do'] = 'split'
                        response_message['k_folds'] = k_folds
                elif received_message['what2do'] == 'echo':
                    if (i_reg == len(regs)) and (i_fold > 0):
                        response_message['what2do'] = 'done'
                    elif client_infos['n_samples'].get(self.client_info, 0) == 0: # it means that the server model is new for the client
                        response_message['what2do'] = 'minimize'
                        response_message['model'] = server_model
                        response_message['coef_'] = server_coef_
                        response_message['i_fold'] = i_fold
                    else:
                        response_message['what2do'] = 'echo'
            response_json = json.dumps(response_message, cls = encdec.MessageEncoder) + '\4'
            response_bin = bytes(response_json, encoding="utf-8")
            try:
                self.connection.sendall(response_bin)
            except BaseException as e:
                print('Error Sending Data to the Client: {msg}.\n'.format(msg=e))
        else:
            print('A dictionary is expected to be received from the client but {d_type} received'.format(d_type=type(received_data)))
        thread_lock.release()

    def run(self):
        global i_iter
        try:
            message_json = json.dumps({'what2do':'reset','version':server_model_version,'i_iter':-1}, cls = encdec.MessageEncoder) + '\4'
            message_bin = bytes(message_json, encoding="utf-8")
            self.connection.sendall(message_bin)
        except BaseException as e:
            print("Error Sending Data to the Client: {msg}.\n".format(msg=e))
        print("Running a Thread for the Connection with {client_info}".format(client_info=self.client_info))
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                i_iter = 0
                if self.client_info in open_clients: open_clients.remove(self.client_info)
                for keys, values in client_infos.items():
                    if self.client_info in values: del values[self.client_info]
                    for client_name, n_samples in client_infos['n_samples'].items():
                        client_infos['n_samples'][client_name] = 0
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break
            self.reply(received_data)

def update_server_model(client_info):
    global server_model, server_coef_, client_infos, i_iter, i_fold, i_reg, min_updated_clients, server_model_version
    print('I_ITER:',i_iter,'I_FOLD:',i_fold,'I_REG:',i_reg,'FROM:',client_info,'CONNECTED:',open_clients)
    server_model.intercept_ = client_infos['models'][client_info].intercept_
    if isinstance(server_coef_, pd.Series):
        server_coef_ = server_coef_.reindex(set(server_coef_.index) | set(client_infos['coef_'][client_info].index))
        server_coef_[client_infos['coef_'][client_info].index] = client_infos['coef_'][client_info].values  
    else:
        server_coef_ = client_infos['coef_'][client_info]
    n_updates_from_clients = np.sum([n_samples > 0 for client_name, n_samples in client_infos['n_samples'].items()])
    min_updates = max(min_updated_clients, len(open_clients))
    if min_updates == 1:
        client_infos['n_samples'][client_info] = 0
    elif n_updates_from_clients >= min_updates:
        for client_name, n_samples in client_infos['n_samples'].items():
            if (n_samples > 0) and (client_name != client_info):
                client_infos['n_samples'][client_name] = 0
    server_model_version += 1
    i_iter += 1

#----- INITIALIZATION
server_ip = 'localhost'
server_port = 10000
regs = np.logspace(np.log10(1e-4), np.log10(1e-1), 4)# array with the tested values for regularization strengths
regs - [1e-2,]
k_folds = 10 # number of folds for cross-validation
i_reg = 0 # index in the array of regularization strenths
i_fold = 0 # index of the fold for K-folds cross-validation
i_iter = 0 # index of the minimization step
i_reg_best = None # index of the optimal regularization strength
learning_rate_0 = 1e-2 # starting learning rate
min_updated_clients = 2 # the server model is updated only after receiving at least this number of updates from the clients, in any case all the connected clients need to update before restarting the cycle
max_iter = 100*min_updated_clients # maximum number of minimization steps
confusion_matrix = np.zeros((2,2,k_folds)) # this one is used to store the confusion matrixes during K-folds cross-validation
client_infos = {'n_samples':{}, 'scores':{}, 'models':{}, 'coef_':{}}
open_clients = [] # list of active clients
acc_means = np.empty(len(regs)) # average accuracy over k-folds at different regularization strengths
acc_stds = np.empty(len(regs)) # standard deviation of the accuracy over k-folds at different regularization strengths
server_model = SGDClassifier(loss = 'log', penalty = 'l1', alpha = regs[i_reg], max_iter = 1, learning_rate = 'constant', eta0 = learning_rate_0, early_stopping = False, tol = None)
server_coef_ = False
server_model_version = 0
#----- END: INITIALIZATION

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
soc.bind((server_ip, server_port))
soc.listen(1)
print("Socket is Listening for Connections...")

while True:
    try:
        connection, client_info = soc.accept()
        print("New Connection from: {client_info}".format(client_info=client_info))
        i_iter = 0
        open_clients.append(client_info)
        socket_thread = SocketThread(connection=connection, client_info=client_info)
        socket_thread.start()
    except:
        soc.close()
        if soc.client_info in open_clients: open_clients.remove(soc.client_info)
        print("(Timeout) Socket Closed Because no Connections Received\n")
        break
