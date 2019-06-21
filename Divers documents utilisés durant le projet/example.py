# In[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers
from deux import *  # Les fonctions que j'ai fait à côté
from plotFile import *
from btk import btk # BTK
from array import *
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
from dictionnaire import *


# Charge les données dans acq
files = allFiles('Sub_DB_Checked\ITW')

# In[3]

capteurDico, genreDico = dicoCapteurGenre()
eventLabel, eventContext = "Foot_Off_GS", "Right"

capteurSet = capteurDico[eventLabel][eventContext]
genre = genreDico[eventLabel][eventContext]

trainingSet, testingSet = selectWithExistingEvent(files, eventLabel, eventContext)
print('On a ', len(trainingSet), 'trainings et ', len(testingSet), 'tests')


erreur = []
erreurFinal = []
weight = weightCreation(capteurSet)
j = 0

moyenne = {}
for capteur in capteurSet:
    moyenne[capteur] = predictDist(trainingSet, eventLabel, eventContext, capteur, genre[capteur], 0.6)    # Ecart moyen entre l'extremum et l'event


for acr in trainingSet:

    j += 1

    n_events = acr.GetEventNumber()
    event_frames = [acr.GetEvent(event).GetFrame() for event in range(n_events)]

    # On récupère la frame de l'event - Utile pour les tests après
    even, evenFrame = closestEvent(acr, acr.GetEvents(), eventLabel, eventContext)
    # print('Exact :', evenFrame)

    # Prédiction des events dans le fichier acr
    predi = calcPredic(acr, moyenne, capteurSet, genre)

    arrayPredi, arrayWeight = np.asarray(dic2mat(predi, capteurSet)) , dic2mat(weight, capteurSet)        # Transformation en array
    meanPrediction = np.round(np.dot(arrayWeight, arrayPredi))         # On applique les poids pour faire la moyenne pondérée
    erreur.append(np.min(np.abs(meanPrediction - evenFrame)))   # On trouve l'erreur (qui est la distance entre l'event et notre prediction correspondante)

    # Mise à jour des poids
    weight = weightUpDate(arrayPredi, evenFrame, weight, capteurSet, j)
# En sortant, on obtient les poids finaux

print('Erreur :', erreur)

erreurFinal = test(testingSet, eventLabel, eventContext, moyenne, weight, capteurSet, genre)

print('Erreur final :', erreurFinal)
print('Poids finaux :', weight)

# # plotChoice(acplus)

# GUIplot(files)
