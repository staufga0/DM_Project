# In[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers
from file import *  # Les fonctions que j'ai fait à côté
from btk import btk # BTK
from array import *
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
# import messageBox

# In[1]
# Charge les données dans acq
files = allFiles('Sub_DB_Checked\ITW')

acq = files[5]
# acq = initial('C:\\Users\Windows\Desktop\BTK\ITW_00613_20100526_04.c3d')
# In[2]
# Infos utiles
n_frames, first_frame, last_frame = frameData(acq)

# On récupérer tous les évènements
AllEvents = acq.GetEvents()

# On met dans un array les frames de tous les events
n_events = acq.GetEventNumber()  # Nombre d'évènements
event_frames = [acq.GetEvent(event).GetFrame() for event in range(n_events)]

metadata = acq.GetMetaData()
point_labels = metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString()
point_labels = np.array(point_labels)
printName(point_labels, globals())


# Partie concernant les plots
axe = 2
axe2 = 2
element = 'HEE'     # Position du cours
droite = 'R' + element        # Partie droite
gauche = 'L' + element
droite = 'T10'
gauche = "C7"

# On récupère les data droite et gauche, selon les axes
dataR = np.array(acq.GetPoint(droite).GetValues()[:, axe])
dataL = np.array(acq.GetPoint(gauche).GetValues()[:, axe2])

# Min et max droite et droite
MinR, MaxR = minLocal(dataR), maxLocal(dataR)
MinL, MaxL = minLocal(dataL), maxLocal(dataL)

# Event le plus proche du début, avec le label (strike / off) et le context (droite / gauche) donné
evenClose, frameClose = closestEvent(dataR[:], AllEvents, label='Foot_Off_GS', context='Left')

positionExtrem, distance, indexMinimDist = closestExtrem(frameClose, MinR)
positionExtrem2, distance2, indexMinimDist2 = closestExtrem(positionExtrem, MinL)

ToutEvent = [['Foot_Off_GS', 'Left'], ['Foot_Strike_GS', 'Left'], ['Foot_Off_GS', 'Right'], ['Foot_Strike_GS', 'Right']]

# for minim in Min:
#     addEvent(acq, 'Foot_Strike_GS', 'Right', int(minim - distance))

# Contient les prédictions
nextFrame = []

# TODO : trouve un min sur 2
# parite = indexMinimDist % 2
# autre = Min[parite::2]

minMinR = findMinMin(dataR, MinR)
minMinL = findMinMin(dataL, MinL)

# printName(minMinR, globals())


# # Fait les prédictions en utilisant les points suivants
# for posi in ['LWRB', 'LSHO', 'LPSI', 'STRN']:
#     dataTest = np.array(acq.GetPoint(posi).GetValues()[:, 2])   # On charge les data
#     evenClose, frameClose = closestEvent(dataTest, AllEvents, 'Foot_Off_GS', 'Left')    # On cherche un évènement
#     if evenClose != 0:
#         MinTest = np.ravel(minLocal(dataTest))
#         positionExtrem, distance, indexMinimDist = closestExtrem(frameClose, MinTest)
#         nextFrame.append(MinTest[indexMinimDist+2] - distance)   # On ajoute une prédiction
#
# printName(nextFrame, globals())
# print(np.round(np.mean(nextFrame)))
#
# # On rajoute l'évènement
# addEvent(acq, 'Foot_Off_GS', 'Left', int(np.round(np.mean(nextFrame))))

# On ouvre une fenêtre
figure = plt.figure(figsize=(8,6))

# Plot part
ax = plt.subplot(2, 1, 1)
ax.plot(np.array(range(first_frame, last_frame + 1)), dataR, 'k')
ax.plot(MinR, dataR[MinR], 'o b')
ax.plot(MaxR, dataR[MaxR], 'o', color='purple')
# ax.plot(MinR[indexMinimDist], dataR[MinR[indexMinimDist]], 'o r')
ax.plot(minMinR, dataR[minMinR], 'o', color='orange')
# ax.plot(autre, dataR[autre], 'o', color='brown')
plt = plotEvent(acq, ax)
plt.title(" Position = {} - axis = {}".format(droite, axe))
ax = plt.subplot(2, 1, 2)
ax.plot(np.array(range(first_frame, last_frame + 1)), dataL, 'k')
ax.plot(MinL, dataL[MinL], 'o b')
ax.plot(MaxL, dataL[MaxL], 'o', color='purple')
ax.plot(minMinL, dataL[minMinL], 'o', color='orange')
# ax.plot(MinL[indexMinimDist2], dataL[MinL[indexMinimDist2]], 'o r')
plt = plotEvent(acq, ax)
ax.set_xlabel(" Position = {} - axis = {}".format(gauche, axe2))
plt.show(block = False)



GUIplot(acq, point_labels)
