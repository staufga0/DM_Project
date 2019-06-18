from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers
from file import *  # Les fonctions que j'ai fait à côté
from btk import btk # BTK
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt

from network import Neural_Network

import torch
import torch.nn as nn

tournus = 4


label = 'Foot_Strike_GS'
contexte = 'Left'
type = 'CP'

if tournus == 1:
    label = 'Foot_Off_GS'
elif tournus == 2:
    contexte = 'Right'
elif tournus == 3:
    contexte = 'Right'
    label = 'Foot_Off_GS'

print('label: ', label, ' contexte: ', contexte, ' type: ', type)


path = 'Sub_DB_Checked'+'\\'+ type
axe = 2

files = allFiles(path)
#GUIplot(files)

for t in np.arange(0, 10, 2):
    print('-----------------------------------t = ', t)
    for lab in ['Foot_Strike_GS','Foot_Off_GS']:
        for cont in ['Left', 'Right']:
            train, test = selectWithExistingEvent(files, lab, cont)
            #print('number of testing instances: ', len(test))
            #print('number of training instances: ',len(train))
            nnOL = Neural_Network(lab,cont,type,5,t)
            nnOL.train(train)
            err = nnOL.test(test)
            print('actual testing error: ', err)
            print('label: ', lab, ' contexte: ', cont, ' type: ', type, ' err on training: ',  np.abs(err).mean())






'''


train, test = selectWithExistingEvent(files, label, contexte)
print('number of testing instances: ', len(test))
print('number of training instances: ',len(train))

'''
'''

nnOL = Neural_Network(label,contexte,type)
nnOL.train(train)
err = nnOL.test(test)
print('actual testing error: ', err)
print( 'mean: ', np.abs(err).mean())

for p in nnOL.model.parameters():
    print(p)
'''
'''
v_batch = int(len(train)/4+0.5)    #used for cross validation using 1/4 of the training set each time
if v_batch == 0:v_batch =1

#start finding best value for hyper parameter h = number of hidden layers, t = tolerance to add to a step
min = 100
best =  Neural_Network(label,contexte,type)
bh=0
berr=[]
for h in np.arange(1, 21, 1):
    print('-----------------------------------h = ', h)
    err = []
    for n in np.arange(0, len(train),v_batch):
        tr = train
        val = []
        for i in range(v_batch):
            if n+i<len(train):
                tr = list(set(tr)-set([train[n+i]]))
                val.append(train[n+i])
        nnOL = Neural_Network(label,contexte,type, h)
        nnOL.setPatch(True)
        nnOL.train(tr)
        err = np.append(err, nnOL.test(val))

    if np.abs(err).mean()<min:
        berr= err
        print(berr)
        min = np.abs(err).mean()
        best = nnOL
        bh=h
print(min)
print('best error on validation Data:', berr)
print('bh = ', bh)

nnOL = Neural_Network(label,contexte,type, bh)
nnOL.setPatch(True)
nnOL.train(train)
err = nnOL.test(test)
print('actual testing error: ', err)
print( 'mean: ', np.abs(err).mean())
print('label: ', label, ' contexte: ', contexte, ' type: ', type)





'''
'''

#cross correlation with tolerance as an hyper parameter

v_batch = int(len(train)/4+0.5)    #used for cross validation using 1/4 of the training set each time
if v_batch == 0:v_batch =1

#start finding best value for hyper parameter h = number of hidden layers, t = tolerance to add to a step
min = 100
best =  Neural_Network(label,contexte,type)
bh=0
bt=0
berr=[]
for h in np.arange(1, 12, 1):
    print('-----------------------------------h = ', h)
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

        if np.abs(err).mean()<min:
            berr= err
            print(berr)
            min = np.abs(err).mean()
            best = nnOL
            bh=h
            bt=t
print(min)
print('best error on validation Data:', berr)
print('bh = ', bh)
print('bt = ', bt)

nnOL = Neural_Network(label,contexte,type, bh, bt)
nnOL.setPatch(True)
nnOL.train(train)
err = nnOL.test(test)
print('actual testing error: ', err)
print( 'mean: ', np.abs(err).mean())
print('label: ', label, ' contexte: ', contexte, ' type: ', type)

'''
'''
for v in train:
    tr=train.remove(v)

nnOL = Neural_Network('Foot_Off_GS','Left','ITW', 20, 5, 0.5)
nnOL.train(train)
err = nnOL.test(validation)
print('validation error: ', err)

min = 100
best =  Neural_Network('Foot_Off_GS','Left','ITW')
bh=5
bt=0
berr=[]
for h in np.arange(1, 7, 1):
    print('-----------------------------------h = ', h)
    for t in np.arange(0, 11, 1):
        print('-----------------------------------t = ', t)

        nnOL = Neural_Network('Foot_Off_GS','Left','ITW', h, t)
        nnOL.train(train)
        err = nnOL.test(validation)
        if np.abs(err).mean()<min:
            berr= err
            min = np.abs(err).mean()
            best = nnOL
            bh=h
            bt=t

print(min)
print('best error on validation Data:', berr)
print('bh = ', bh)
print('bt = ', bt)

'''
'''

train, test = selectWithExistingEvent(files, 'Foot_Off_GS', 'Left')
nnOL = Neural_Network('Foot_Off_GS','Left','ITW')
nnOL.train(train)
err = nnOL.test(test)
print('testing error: ', err)
nnOL.addPredictedEvent(acq)

#GUIplot([acq])

print( '-----------------------------------Foot_Strike_GS Left')

train, test = selectWithExistingEvent(files, 'Foot_Strike_GS', 'Left')

nnSL = Neural_Network('Foot_Strike_GS', 'Left','ITW')
nnSL.train(train)
err = nnSL.test(test)
print('testing error: ', err)
nnSL.addPredictedEvent(acq)

print( '-----------------------------------Foot_Off_GS Right')
train, test = selectWithExistingEvent(files, 'Foot_Off_GS', 'Right')

nnOR = Neural_Network('Foot_Off_GS','Right','ITW')
nnOR.train(train)
err = nnOR.test(test)
print('testing error: ', err)
nnOR.addPredictedEvent(acq)

print( '-----------------------------------Foot_Strike_GS Right')
train, test = selectWithExistingEvent(files, 'Foot_Strike_GS', 'Right')

nnSR = Neural_Network('Foot_Strike_GS','Right','ITW')
nnSR.train(train)
err = nnSR.test(test)
print('testing error: ', err)
nnSR.addPredictedEvent(acq)

save(acq, 'test2.c3d')
#GUIplot([acq])
for numevent in range(acq.GetEventNumber()):
    event = acq.GetEvent(numevent)
    print('label: ', event.GetLabel(), ' context: ', event.GetContext(), ' frame: ', event.GetFrame())


acq = initial('test2.c3d')
print('---------------after save and reload')
for numevent in range(acq.GetEventNumber()):
    event = acq.GetEvent(numevent)
    print('label: ', event.GetLabel(), ' context: ', event.GetContext(), ' frame: ', event.GetFrame())

#GUIplot([acq])
#print('number of testing instances: ', len(test))
#print('number of training instances: ',len(train))

'''

'''
acq = files[9]

metadata = acq.GetMetaData()
point_labels = list(metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString())

point_labels = ['RFHD','LFHD','LBHD','RSHO','RASI','RPSI','LPSI','RHEE', 'STRN','CLAV','T10','C7']
min_max = np.array([])
print(point_labels)
for l in point_labels:
    data = np.array(acq.GetPoint(l.strip()).GetValues()[:, 2])

    Min, Max = minLocal(data), maxLocal(data)
    #plt.plot(Min, np.zeros(Min.shape[0]), 'o b')
    #plt.legend('minimum for capteur ', l)

    #min_max = np.append(min_max,Min)
    min_max = np.append(min_max,Max)

#plt.plot(min_max, np.zeros(min_max.shape[0]), 'o b')
#plt.ylabel('some numbers')
#plt.show()
min_max = np.sort(min_max)
isole = []
for i in range(min_max.shape[0]-1):
    if i != 0:
        if(min_max[i]-min_max[i-1]>2 and min_max[i+1]-min_max[i]>2):
            isole.append(min_max[i])

print(isole)

capteur_label = []
for l in point_labels:

    data = np.array(acq.GetPoint(l.strip()).GetValues()[:, 2])

    Min, Max = minLocal(data), maxLocal(data)
    if(len(set(Min.tolist())& set(isole)) !=0 ): capteur_label.append('min '+l)
    if(len(set(Max.tolist())& set(isole)) !=0 ): capteur_label.append('max '+l)
print(capteur_label)

'''
