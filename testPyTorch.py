from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers
from file import *  # Les fonctions que j'ai fait à côté
from btk import btk # BTK
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt

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


train, test = selectWithExistingEvent(files, 'Foot_Off_GS', 'Left')

#print('number of testing instances: ', len(test))
#print('number of training instances: ',len(train))

acq = train[1]
'''
metadata = acq.GetMetaData()
point_labels = list(metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString())
min_max = np.array([])
print(point_labels)
for l in point_labels:

    data = np.array(acq.GetPoint(l.strip()).GetValues()[:, 2])
    Min, Max = minLocal(data), maxLocal(data)
    np.append(min_max,Min)
    np.append(min_max,Max)
print(min_max)

figure = plt.figure(figsize=(8,6))
ax = plt.subplot()

ax.plot(min_max, np.zeros(min_max.shape[0]), 'o b')

plt.show(block = False)
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
Xtrain = []
Ytrain = []
for acq in np.append(train,test):
    #print('event--------------------------')
    n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
    for numevent in range(n_events):            # On parcours les indices des évènements
        event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
        if event.GetLabel() == 'Foot_Off_GS' and event.GetContext() == 'Left':
            event_frame = event.GetFrame()
            Ytrain.append(event_frame)
            start_step, end_step = selectStep(acq, event)
            #print('starting frame:', start_step)
            #print('event frame: ', event_frame)
            #print('end frame: ', end_step)
            shapedData = shapeStepDataITW(acq, start_step, end_step)
            Xtrain.append(shapedData)
            #np.append(shaped_train, [shapedData], axis=0)
            #print(shapedData)
            #print('shape of Data: ', shapedData.shape)

Xtrain =np.array(Xtrain)
Ytrain = np.array(Ytrain)

print(Ytrain)
y = torch.from_numpy(Ytrain.reshape(Ytrain.shape[0],1)).float()
print(y)
#------------------------------------------------------------------------------------------------training part

n_in, n_h, n_out, batch_size = Xtrain.shape[1], 20, 1, Xtrain.shape[0]

x = torch.from_numpy(Xtrain).float()
print(x.shape)
#x = torch.randn(batch_size, n_in)
y = torch.from_numpy(Ytrain.reshape(Ytrain.shape[0],1)).float()
print(y.shape)

model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     #nn.Sigmoid()
                     )

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # Forward Propagation
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()
    if epoch == 99 : print(y_pred)
print(y)
