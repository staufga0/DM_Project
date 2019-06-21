# In[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers
from functionFile import *  # Les fonctions que j'ai fait à côté
from btk import btk # BTK
from array import *
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
# import messageBox



def plotChoice(acq):
    import matplotlib   # Pour les plots
    import matplotlib.pyplot as plt
    # Partie concernant les plots

    AllEvents = acq.GetEvents()
    n_frames, first_frame, last_frame = frameData(acq)

    axe = 2
    axe2 = 2
    element = 'ANK'     # Position du cours
    droite = 'R' + element        # Partie droite
    gauche = 'L' + element
    # droite = 'T10'
    # gauche = "C7"


    # On récupère les data droite et gauche, selon les axes
    dataR = np.array(acq.GetPoint(droite).GetValues()[:, axe])
    dataL = np.array(acq.GetPoint(gauche).GetValues()[:, axe2])

    # Les milieux
    taux = 0.75
    milieuR = int(np.round((np.max(dataR)*taux + np.min(dataR)*(1-taux))))
    milieuL = int(np.round((np.max(dataL)*taux + np.min(dataL)*(1-taux))))

    # Min et max droite et droite
    MinR, MaxR = minLocal(dataR), maxLocal(dataR)
    MinL, MaxL = minLocal(dataL), maxLocal(dataL)

    # print(MinR[1:] - MinR[:-1])

    minMinR = findMinMin(dataR, MinR)
    minMinL = findMinMin(dataL, MinL)

    # Event le plus proche du début, avec le label (strike / off) et le context (droite / gauche) donné
    # evenClose, frameClose = closestEvent(dataR[:], AllEvents, label='Foot_Off_GS', context='Left')

    # positionExtrem, distance, indexMinimDist = closestExtrem(frameClose, MinR)
    # positionExtrem2, distance2, indexMinimDist2 = closestExtrem(positionExtrem, MinL)

    # On ouvre une fenêtre
    figure = plt.figure(figsize=(8,6))

    # Plot part
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.array(range(first_frame-1, last_frame)), dataR, 'k')
    ax.plot(MinR, dataR[MinR], 'o b')
    ax.plot(MaxR, dataR[MaxR], 'o', color='purple')
    ax.axhline(y = milieuR)
    # ax.plot(MinR[indexMinimDist], dataR[MinR[indexMinimDist]], 'o r')
    ax.plot(minMinR, dataR[minMinR], 'o', color='orange')
    # ax.plot(autre, dataR[autre], 'o', color='brown')
    ax = plotEvent(acq, ax)
    plt.title(" Position = {} - axis = {}".format(droite, axe))

    ax = plt.subplot(2, 1, 2)
    ax.plot(np.array(range(first_frame, last_frame + 1)), dataL, 'k')
    ax.plot(MinL, dataL[MinL], 'o b')
    ax.plot(MaxL, dataL[MaxL], 'o', color='purple')
    ax.plot(minMinL, dataL[minMinL], 'o', color='orange')
    ax.axhline(y = milieuL)
    # ax.plot(MinL[indexMinimDist2], dataL[MinL[indexMinimDist2]], 'o r')
    ax = plotEvent(acq, ax)
    ax.set_xlabel(" Position = {} - axis = {}".format(gauche, axe2))
    plt.show(block=True)
