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

path = 'Sub_DB_Checked\ITW'
axe = 2

files = []

# Pour trouver tous les fichiers .c3d
for r, d, f in os.walk(path):
    for file in f:
        if '.c3d' in file:
            files.append(initial(os.path.join(r, file)))


train, test = selectWithExistingEvent(files, 'Foot_Off_GS', 'Left')

print(len(test))
print(len(train))

acq = train[1]


n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
for numevent in range(n_events):            # On parcours les indices des évènements
    event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
    if event.GetLabel() == 'Foot_Off_GS' and event.GetContext() == 'Left':
        event_frame = event.GetFrame()
        start_step, end_step = selectionnePas(acq, event)
        print(start_step)
        print(event_frame)
        print(end_step)


for acq in train:
    print('event--------------------------')
    n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
    for numevent in range(n_events):            # On parcours les indices des évènements
        event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
        if event.GetLabel() == 'Foot_Off_GS' and event.GetContext() == 'Left':
            event_frame = event.GetFrame()
            start_step, end_step = selectionnePas(acq, event)
            print(start_step)
            print(event_frame)
            print(end_step)
