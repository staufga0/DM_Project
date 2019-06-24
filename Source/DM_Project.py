 # In[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers

from GUIplot import *
from plotFile import *
from btk import btk # BTK
from array import *
import matplotlib   # Pour les plots
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
from functionFile import *
from dictionnaire import *

type = '\ITW'
path = 'Sub_DB_Checked' + type

# Charge les données dans acq
files = allFiles(path)

# In[3]

capteurDico, genreDico = dicoCapteurGenre(type)
eventLabel, eventContext = "Foot_Off_GS", "Left"
# Red = 'Left'    -        Green = 'Right'

evenLabel_Set = ["Foot_Strike_GS", "Foot_Off_GS"]
eventContext_Set = ["Left", "Right"]
acfin = files[-1]
acfinal = files[-2]
n_events = acfinal.GetEventNumber()

for num in range(n_events):
    acfinal.RemoveEvent(0)

nombre_ite = 10


for eventContext in eventContext_Set:
    for eventLabel in evenLabel_Set:
        print('\n\nLabel : ', eventLabel, ' - Context :', eventContext)

        for p in range(nombre_ite):

            # On divise les fichiers entre Training et Testing
            trainingSet, testingSet = selectWithExistingEvent(files, eventLabel, eventContext)
            # print('On a ', len(trainingSet), 'trainings et ', len(testingSet), 'tests')

            # Les capteurs qu'on choisit et si on utilise des min ou des max
            capteurSet = capteurDico[eventLabel][eventContext]
            genre = genreDico[eventLabel][eventContext]

            # Initialisation des variables
            erreur, erreurFinal = [], []
            numTest = 0
            weight = weightCreation(capteurSet)

            # Calcul des distances moyennes entre un event et un extremum
            moyenne = {capteur : mean_distance(trainingSet, eventLabel, eventContext, capteur, genre[capteur], 0.6) for capteur in capteurSet}   # Ecart moyen entre l'extremum et l'event

            # On parcourt tous les fichiers d'entrainements (trainingSet)
            for acr in trainingSet:

                # Numéro du test (combien de test on a déjà fait)
                numTest += 1

                # On récupère la frame de l'event - Utile pour les tests après (et calculer l'erreur)
                even, evenFrame = closestEvent(acr, acr.GetEvents(), eventLabel, eventContext)

                # Dico où chaque capteur fait une prédiction des events
                predi_dic = calcPredic(acr, moyenne, capteurSet, genre, type)



                # Renvoie les prédictions de chaque capteur, et l'erreur totale
                prediction, erreur = calculErreur(predi_dic, capteurSet, evenFrame, weight, erreur)

                # Mise à jour des poids
                weight = weightUpDate(prediction, evenFrame, weight, capteurSet, numTest)

            # En sortant, on obtient les poids finaux

            # print('**Erreur :', (erreur))
            # print('Moyenne :', np.mean(erreur))
            # print('Poids finaux :', weight)

            # Erreur de chaque testing file
        erreurFinal = test(testingSet, eventLabel, eventContext, moyenne, weight, capteurSet, genre)


        # print('Erreur final :', erreurFinal)
        print('Moyenne :', np.mean(erreurFinal))
        # print('Poids finaux :', weight)



# Si on souhaite ploter un graphe avec les events rajouter, il suffit d'enlever les commentaires des 4 lignes suivants :

# plt.figure(figsize=(9,7))
# guiPlot = plt.subplot()
# guiPlot = plotPosi(acfinal, 'RANK', 'z', guiPlot)
# plt.show(block=True)
# GUIplot(trainingSet)



# writer = btk.btkAcquisitionFileWriter()
# writer.SetInput(acfin)
# writer.SetFilename('newFile.c3d')
# writer.Update()
