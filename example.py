# In[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers
from file import *  # Les fonctions que j'ai fait à côté
from btk import btk # BTK
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt

# In[1]
# Charge les données dans acq
acc = [initial("Sub_DB_Checked\ITW\ITW_01758_20120523_23.c3d")]
acc.append(initial("Sub_DB_Checked\ITW\ITW_01830_20121008_05.c3d"))
acc.append(initial("Sub_DB_Checked\ITW\ITW_01888_20140319_14.c3d"))
acc.append(initial("Sub_DB_Checked\ITW\ITW_01889_20130114_14.c3d"))

acq = acc[0]

# Le dossier où j'ai le projet
path = 'Sub_DB_Checked\ITW'

files = []

# Pour trouver tous les fichiers .c3d
for r, d, f in os.walk(path):
    for file in f:
        if '.c3d' in file:
            files.append(initial(os.path.join(r, file)))

acq = files[3]
# In[2]
# Infos utiles
n_frames, first_frame, last_frame = frameData(acq)
print(n_frames)

# metadata (utile après pour récupérer des données, comme les labels)
metadata = acq.GetMetaData()

# Simple test pour essayer la fonction
event, label, context, frame = eventInfo(acq, 2)

# On récupérer tous les évènements
AllEvents = acq.GetEvents()

# On prend tous les points (mais variable par utilisé après, donc pas nécessaire)
point_labels = metadata.FindChild("POINT").value().FindChild(
    "LABELS").value().GetInfo().ToString()

# On met dans un array les frames de tous les events
n_events = acq.GetEventNumber()  # Nombre d'évènements
event_frames = [acq.GetEvent(event).GetFrame() for event in range(n_events)]

# In[3]
start_frame = event_frames[0] - first_frame         # Frame du premier évènement
end_frame = event_frames[-1] - first_frame          # Frame du dernier évènement

# Partie concernant les plots
axe = 2
axe2 = 2
element = 'HEE'     # Position du cours
droite = 'R' + element        # Partie droite
gauche = 'L' + element
# position = 'LWRB'

# On récupère les data droite et gauche, selon les axes
dataR = np.array(acq.GetPoint(droite).GetValues()[:, axe])
dataL = np.array(acq.GetPoint(gauche).GetValues()[:, axe2])

# Min et max droite
indMin = np.ravel(minLocal(dataR[:]))
indMaxR = np.ravel(maxLocal(dataR[:]))

# Min et max gauche
indMinL = np.ravel(minLocal(dataL[:]))
indMaxL = np.ravel(maxLocal(dataL[:]))

# Event le plus proche du début, avec le context et le label donné
evenClose, frameClose = closestEvent(dataR[:], AllEvents, 'Foot_Off_GS', 'Right')

positionExtrem, distance, indexMinimDist = closestExtrem(frameClose, indMin)
positionExtrem2, distance2, indexMinimDist2 = closestExtrem(positionExtrem, indMinL)

# for minim in indMin:
#     addEvent(acq, 'Foot_Strike_GS', 'Right', int(minim - distance))

# Contient les prédictions
nextFrame = []

# Fait les prédictions en utilisant les points suivants
for posi in ['LELB', 'LWRB', 'LSHO', 'LPSI', 'STRN']:
    dataTest = np.array(acq.GetPoint(posi).GetValues()[:, 2])   # On charge les data
    evenClose, frameClose = closestEvent(dataTest, AllEvents, 'Foot_Off_GS', 'Left')    # On cherche un évènement
    if evenClose != 0:
        indMinTest = np.ravel(minLocal(dataTest))
        positionExtrem, distance, indexMinimDist = closestExtrem(frameClose, indMinTest)
        nextFrame.append(indMinTest[indexMinimDist+2] - distance)   # On ajoute une prédiction

printName(nextFrame, globals())
print(np.round(np.mean(nextFrame)))

# On rajoute l'évènement
addEvent(acq, 'Foot_Off_GS', 'Left', int(np.round(np.mean(nextFrame))))

# On ouvre une fenêtre
ax = plt.figure(figsize=(8,6))

# Plot part
ax = plt.subplot(2, 1, 1)
ax.plot(np.array(range(first_frame, last_frame + 1)), dataR[:], 'k')
ax.plot(indMin, dataR[indMin], 'o b')
ax.plot(indMaxR, dataR[indMaxR], 'o', color='purple')
# ax.plot(indMin[indexMinimDist], dataR[indMin[indexMinimDist]], 'o r')
# ax.plot(indMin[indexMinimDist+2], dataR[indMin[indexMinimDist+2]], 'o r')
plt = plotEvent(acq, ax)
plt.title(" Position = {} - axis = {}".format(droite, axe))

ax = plt.subplot(2, 1, 2)
ax.plot(np.array(range(first_frame, last_frame + 1)), dataL[:], 'k')
ax.plot(indMinL, dataL[indMinL], 'o b')
ax.plot(indMaxL, dataL[indMaxL], 'o', color='purple')
# ax.plot(indMinL[indexMinimDist2], dataL[indMinL[indexMinimDist2]], 'o r')
plt = plotEvent(acq, ax)
ax.set_xlabel(" Position = {} - axis = {}".format(gauche, axe2))
plt.show()
