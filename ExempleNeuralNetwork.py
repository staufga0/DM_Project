from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers

import sys
sys.path.append('./Source')

from file import *  # Les fonctions que j'ai fait à côté
from btk import btk # BTK
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt

from network import Neural_Network

import torch
import torch.nn as nn

# apply the cross validation method two determine the best hyper parameters
def crossValidation(label, contexte, type):

    path = 'Sub_DB_Checked'+'\\'+ type
    files = allFiles(path)
    train, test = selectWithExistingEvent(files, label, contexte)
    v_batch = int(len(train)/4+0.5)    #used for cross validation using 1/4 of the training set each time
    if v_batch == 0:v_batch =1

    #start finding best value for hyper parameter h = number of hidden layers, t = tolerance to add to a step
    min = 100
    best =  Neural_Network(label,contexte,type)
    bh=0                    # the best number of neurone
    bt=0                    # the best tolerance for the boundaries of the step
    berr=[]
    for h in np.arange(1, 12, 1):
        print('--------------------------------------------h = ', h)
        for t in np.arange(0, 10, 2):
            print('-----------------------------------t = ', t)
            err = []
            for n in np.arange(0, len(train),v_batch):
                tr = train
                val = []
                for i in range(v_batch):
                    if n+i<len(train):
                        tr = list(set(tr)-set([train[n+i]]))
                        val.append(train[n+i])
                nnOL = Neural_Network(label,contexte,type, h, t)
                nnOL.setPatch(True)
                nnOL.train(tr)
                err = np.append(err, nnOL.test(val))
            print('error = ', err)
            if np.abs(err).mean()<min:
                berr= err
                #print(berr)
                min = np.abs(err).mean()
                best = nnOL
                bh=h
                bt=t
    #print(min)
    #print('best error on validation Data:', berr)
    print('bh = ', bh)
    print('bt = ', bt)

    bnn = Neural_Network(label,contexte,type, bh, bt)
    bnn.train(train)
    err = bnn.test(test)
    print('actual testing error: ', err)
    print( 'mean: ', np.abs(err).mean())
    print('label: ', label, ' contexte: ', contexte, ' type: ', type)

    return bh, bt, bnn


# train the 12 networks on all the datas using the hyper parameters found in cross validation
def trainAllNetwork():

    #The 4 ITW networks
    ITW = []
    filesITW = allFiles('Sub_DB_Checked\ITW')
    type = 'ITW'

    label = 'Foot_Strike_GS'
    contexte = 'Left'
    train, test = selectWithExistingEvent(filesITW, label, contexte)
    network = Neural_Network(label,contexte,type, 1, 8)
    network.train(train+test)
    ITW.append(network)

    label = 'Foot_Off_GS'
    contexte = 'Left'
    train, test = selectWithExistingEvent(filesITW, label, contexte)
    network = Neural_Network(label,contexte,type, 10, 0)
    network.train(train+test)
    ITW.append(network)

    label = 'Foot_Off_GS'
    contexte = 'Right'
    train, test = selectWithExistingEvent(filesITW, label, contexte)
    network = Neural_Network(label,contexte,type, 6, 0)
    network.train(train+test)
    ITW.append(network)

    label = 'Foot_Strike_GS'
    contexte = 'Right'
    train, test = selectWithExistingEvent(filesITW, label, contexte)
    network = Neural_Network(label,contexte,type, 1, 8)
    network.train(train+test)
    ITW.append(network)

    #The 4 FD networks
    FD = []
    filesFD = allFiles('Sub_DB_Checked\FD')
    type = 'FD'

    label = 'Foot_Strike_GS'
    contexte = 'Left'
    train, test = selectWithExistingEvent(filesFD, label, contexte)
    network = Neural_Network(label,contexte,type, 1, 4)
    network.train(train+test)
    FD.append(network)

    label = 'Foot_Off_GS'
    contexte = 'Left'
    train, test = selectWithExistingEvent(filesFD, label, contexte)
    network = Neural_Network(label,contexte,type, 8, 2)
    network.train(train+test)
    FD.append(network)

    label = 'Foot_Off_GS'
    contexte = 'Right'
    train, test = selectWithExistingEvent(filesFD, label, contexte)
    network = Neural_Network(label,contexte,type, 8, 4)
    network.train(train+test)
    FD.append(network)

    label = 'Foot_Strike_GS'
    contexte = 'Right'
    train, test = selectWithExistingEvent(filesFD, label, contexte)
    network = Neural_Network(label,contexte,type, 11, 8)
    network.train(train+test)
    FD.append(network)


    #The 4 CP networks
    CP = []
    filesCP = allFiles('Sub_DB_Checked\CP')
    type = 'CP'

    label = 'Foot_Strike_GS'
    contexte = 'Left'
    train, test = selectWithExistingEvent(filesCP, label, contexte)
    network = Neural_Network(label,contexte,type, 1, 0)
    network.train(train+test)
    CP.append(network)

    label = 'Foot_Off_GS'
    contexte = 'Left'
    train, test = selectWithExistingEvent(filesCP, label, contexte)
    network = Neural_Network(label,contexte,type, 5, 8)
    network.train(train+test)
    CP.append(network)

    label = 'Foot_Off_GS'
    contexte = 'Right'
    train, test = selectWithExistingEvent(filesCP, label, contexte)
    network = Neural_Network(label,contexte,type, 1, 0)
    network.train(train+test)
    CP.append(network)

    label = 'Foot_Strike_GS'
    contexte = 'Right'
    train, test = selectWithExistingEvent(filesCP, label, contexte)
    network = Neural_Network(label,contexte,type, 10, 8)
    network.train(train+test)
    CP.append(network)

    return ITW, FD, CP


def predictEvents(files, networks):
    #files contexteains the data on wich we want too predict the n_events
    #networks contexteains the four trained networks corresponding to the four type of n_events
    for n in networks:
        n.addPredictedEvent(files)



'''
# Example of cross-validation :
label = 'Foot_Strike_GS'
contexte = 'Left'
type = 'ITW'
bh, bt, bnn = crossValidation(label, contexte, type)
'''


# Exemple of predicting all the event for the ITW files.
ITW, FD, CP = trainAllNetwork() # we generate the 12 neural networks

filesITW = allFiles('Sub_DB_Checked\ITW') # we load the ITW files
predictEvents(filesITW, ITW) # this function add all the predicted events to all the files
GUIplot(filesITW)
