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

path = 'Sub_DB_Checked\ITW'
axe = 2

files = []

# Pour trouver tous les fichiers .c3d
for r, d, f in os.walk(path):
    for file in f:
        if '.c3d' in file:
            files.append(initial(os.path.join(r, file)))






'''
train, test = selectWithExistingEvent(files, 'Foot_Off_GS', 'Left')
train, test = selectWithExistingEvent(files, 'Foot_Strike_GS', 'Left')
print('number of testing instances: ', len(test))
print('number of training instances: ',len(train))
'''
acq = files[1]
print( '-----------------------------------Foot_Off_GS Left')

#train, test = selectWithExistingEvent(files, 'Foot_Off_GS', 'Left')
train, validation, test = selectWithExistingEventWithValidation(files, 'Foot_Off_GS', 'Left')

nnOL = Neural_Network('Foot_Off_GS','Left','ITW')
nnOL.train(train)
err = nnOL.test(validation)
print('validation error: ', err)

'''
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
GUIplot([acq])
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

'''
n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
for numevent in range(n_events):            # On parcours les indices des évènements
    event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
    if event.GetLabel() == 'Foot_Off_GS' and event.GetContext() == 'Left':
        event_frame = event.GetFrame()
        start_step, end_step = selectStep(acq, event)
        print(start_step)
        print(event_frame)
        print(end_step)
'''
'''
Xtrain = []
Ytrain = []
step_length = []
for acq in train:
    #print('event--------------------------')
    n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
    for numevent in range(n_events):            # On parcours les indices des évènements
        event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
        if event.GetLabel() == 'Foot_Off_GS' and event.GetContext() == 'Left':
            event_frame = event.GetFrame()
            start_step, end_step = selectStep(acq, event)
            step_length.append(end_step-start_step)
            Ytrain.append((event_frame-start_step))
            #print('starting frame:', start_step)
            #print('event frame: ', event_frame)
            #print('end frame: ', end_step)
            shapedData = shapeStepDataITW(acq, start_step, end_step)
            Xtrain.append(shapedData)
            #np.append(shaped_train, [shapedData], axis=0)
            #print(shapedData)
            #print('shape of Data: ', shapedData.shape)

Xtrain =np.array(Xtrain)
#Xtrain = Xtrain/np.(Xtrain)
#print(Xtrain)
step_length = np.array(step_length)
Ytrain = np.array(Ytrain)
Ynorm = Ytrain/step_length

print('moyenne output = ', np.mean(Ytrain))
#------------------------------------------------------------------------------------------------training part

n_in, n_h, n_out, batch_size = Xtrain.shape[1], 40, 1, Xtrain.shape[0]
n_h2=20
n_h3 = 20

x = torch.from_numpy(Xtrain).float()
print(x.shape)
#x = torch.randn(batch_size, n_in)
y = torch.from_numpy(Ynorm.reshape(Ynorm.shape[0],1)).float()
print(y.shape)

model = nn.Sequential(nn.Linear(n_in, n_h, bias=True),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out, bias=True),
                     nn.Sigmoid()
                     )
#criterion = torch.nn.L1Loss()
#criterion = torch.nn.SmoothL1Loss()
criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
y_pred = []
for epoch in range(1000):
    # Forward Propagation
    y_pred = model(x)
    if epoch<5 : print(y_pred)
    # Compute and print loss
    loss = criterion(y_pred, y)
    if epoch % 20 == 0 : print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()
print(y_pred)

print(y_pred-y)
y_prednp = y_pred.detach().numpy().transpose()
result = y_prednp*step_length
print(result)
print(Ytrain)

print(Ytrain-np.vectorize(round)(result))
#print(round(result))
'''
