from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from btk import btk
import os

import matplotlib

import matplotlib.pyplot as plt
from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk

# Label = strike / off ; context = droite / gauche

def filtreExtremum(extrem, originalData):
    if 0 in extrem:
        extrem = extrem[1:]
    if len(originalData)-1 in extrem:
        extrem = extrem[:-1]
    return extrem


def rajoutBout(a, indExtrem):
    indExtrem = np.append(0, indExtrem)
    indExtrem = np.append(indExtrem, len(a)-1)
    return indExtrem

# But : trouvé tous les maximum locaux
# In : un vecteur de taille nx1
# Out : les positions x des max locaux (pas leur valeur y)
def maxLocal(a, bout=0):
    TFarray = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True])   # Rempli un vecteur avec que des False, sauf lorsqu'une donnée dans le vecteur est plus grande que son voisin de droite et de gauche (il met alors True)
    indMax = np.ravel( np.where( TFarray == True ) )    # On récupère les index où il y a les True
    indMax = filtreExtremum(indMax, a)
    if bout == 1:
        indMax = rajoutBout(a, indMax)
    return indMax

# Fonctions en cours, pas encore effective
def semiMaxLocal(a):
    TFarray = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] == a[1:], True])
    indSemiMax = np.where( TFarray == True )
    return indSemiMax

def findMinMin(data, Min):
    minMin = minLocal(data[Min])
    return Min[minMin]

# Pareil que maxLocal, mais pour trouver les minimum locaux
def minLocal(a, bout=0):
    TFarray = np.array(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
    indMin = np.ravel( np.where( TFarray == True ) )
    indMin = filtreExtremum(indMin, a)
    if bout == 1:
        indMin = rajoutBout(a, indMin)
    return indMin

# Dico avec comme clé les labels, et comme valeur un entier
# e.g. : DicoLabels = {"LSHO" = 0, "RSHO" = 1, "RANK" = 2, ...}
# Fonction jamais utilisé pour l'instant
def dicLab(metadata):
    point_labels = metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString()

    dicoLabels = {}
    index = 0
    for lab in point_labels:
        dicoLabels[lab] = index
        index += 1

    return dicoLabels

# Plot les events dans la figure "ax", lignes verticales
# In : acq, qui contient les events ; ax, où on va ploter les lignes verticales
# Out : la nouvelle figure, où on a ploté les lignes
def plotEvent(acq, ax):
    n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
    for numevent in range(n_events):            # On parcours les indices des évènements
        event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
        event_frame = event.GetFrame()          # On récupère la frame où se situe l'évènement
        context = event.GetContext()            # On récupère le context (e.g. Left ou Right)
        label = event.GetLabel()                # On récupère le label (e.g. : Foot_Strike_GS)
        if context == 'Left':                   # Test si c'est le pied gauche
            if label == 'Foot_Strike_GS':       # Test si c'est quand le pied touche le sol
                leftLineStrike = ax.axvline(x = event_frame, color='r', label='Left - Strike', linestyle='--')  # Plot en rouge, avec des tirets
#                ax.legend([leftLineStrike], 'Left - Strike')
            elif label == 'Foot_Off_GS':        # Test si c'est quand le pied ne touche plus le sol
                leftLineOff = ax.axvline(x = event_frame, color='r', label='Left - Off', linestyle='-.')        # Plot en rouge, avec des tirets et des points
        if context == 'Right':                  # Test si c'est le pied droit
            if label == 'Foot_Strike_GS':       # Test si c'est quand le pied touche le sol
                rightLineStrike = ax.axvline(x = event_frame, color='g', label='Righ - Strike', linestyle='--') # Plot en vert, avec des tirets
            elif label == 'Foot_Off_GS':        # Test si c'est quand le pied ne touche plus le sol
                rightLineOff = ax.axvline(x = event_frame, color='g', label='Right - Off', linestyle='-.')      # Plot en rouge, avec des tirets et des points


    # On rajoute la légende
    # S'IL Y A UNE ERREUR, ENLEVER CETTE LIGNE
#    ax.legend((leftLineOff, rightLineStrike, rightLineOff), ('Left - Off', 'Right - Strike', 'Right - Off'))

    return ax

# Selectionne les éléments de files ayant un event de Type labels
# Renvoie en training set (2/3) et un testing set (1/3) constitués de ces éléments.
def selectWithExistingEvent(files, lab, cont):
    eventfiles = []
    for acq in files:
        n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
        for numevent in range(n_events):            # On parcours les indices des évènements
            event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
            if event.GetLabel() == lab and event.GetContext()==cont:                        # Test si c'est le label recherché
                eventfiles.append(acq)
                break
    test = np.random.choice(eventfiles, (len(eventfiles)//3), replace = False)
    train = list(set(eventfiles)-set(test))
    return train, test

# Calcule les frames de départ et de fin du "pas" dans lequel se trouve events
# un "pas" et définit comme la durée entre deux max du talon du coté correpondant à l'event
def selectStep(acq, event):
    if event[0] == 'L':
        capteur = 'LHEE'
    else:
        capteur = 'RHEE'
    taux = 0.5
    data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
    indMax = np.ravel(maxLocal(data))
    seuil = np.mean(np.max(data)*taux + np.min(data)*(1-taux))
    indMax = indMax[data[indMax] > seuil]
    # event_frame = event.GetFrame()
    # for i in range(len(indMax)):
    #     if indMax[i]>event_frame:
    #         start_step = indMax[i-1]+1
    #         end_step = indMax[i]+1
    #         break
    return indMax[0], indMax[1]

# shape
def shapeStepDataITW(acq, start_step, end_step):
    step = []
    type = []
    resume ={}
    #capteurs = ['RFHD', 'LFHD','RPSI','LPSI', 'RTOE','LTOE', 'STRN', 'CLAV','T10','C7']
    #capteurs = ['RFHD', 'LFHD','RPSI','LPSI','RASI','LASI','RWRA', 'LWRA', 'STRN', 'CLAV','T10','C7']
    capteurs = ['STRN', 'CLAV','T10']
    for capteur in capteurs:
        data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
        indMax = np.ravel(maxLocal(data))
        cnt = 0
        for i in indMax:
            if start_step <= i and  i<= end_step:
                step.append(i)
                type.append(capteur + ' max')
                cnt += 1
        resume[capteur + ' max'] = cnt
        cnt = 0
        indMin = np.ravel(minLocal(data))
        for i in indMin:
            if start_step <= i and  i<= end_step:
                step.append(i)
                type.append(capteur + ' min')
                cnt+=1
        resume[capteur + ' min'] = cnt
    #print(type)
    #print(resume)
    if(len(step)!=12):print('ARRETER TOUT !!!!! ', len(step), resume)
    return np.array(step)


# But : Récupérer les données
# In : path des données (Attention : le chemin commence de là où est le fichier)
# Out : les données
def initial(pathFile):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(pathFile)
    reader.Update()
    acq = reader.GetOutput()
    return acq

def allFiles(path):
    files = []

    # Pour trouver tous les fichiers .c3d
    for r, d, f in os.walk(path):
        for file in f:
            if '.c3d' in file:
                files.append(initial(os.path.join(r, file)))
    return files

# But : avoir des infos à propos des frames de "acq"
# In : les données acq
# Out : nombres de frames, numéro de la 1ère frame, numéro de la dernière frame
def frameData(acq):
# get some parameters
    n_frames = acq.GetPointFrameNumber()  # give the number of frames
    first_frame = acq.GetFirstFrame()
    last_frame = acq.GetLastFrame()
    return n_frames, first_frame, last_frame


# But : créer un nouvel évènement
# Un nouvel évènement est caractérisé par un label, un context, et un numéro de frame
# In : les données "acq", un label, un context, et une frame
def addEvent(acq, label, context, frameNumber):
    newEvent = btk.btkEvent()           # Créer un nouvel évènement vide
    newEvent.SetLabel(label)            # Met un label
    newEvent.SetContext(context)        # Met un context
    newEvent.SetFrame(frameNumber)      # Met la positoin, la frame
    acq.AppendEvent(newEvent)           # Rajoute l'évènement parmi tous les autres évènements

# But : équivalent à print('obj = ', obj)
# Pas nécessaire pour le projet
def printName(obj, namespace):
    nom = [name for name in namespace if namespace[name] is obj]
    print(nom[0],' = ', obj)

# But : Avoir toutes les infos d'un évènements
# In : les données "acq", et le numéro de l'évènement
# Out : l'évènement, le label, le context, et le num de la frame
def eventInfo(acq, numEvent):
    event = acq.GetEvent(0) # extract the first event of the aquisition
    label = event.GetLabel() # return a string representing the Label
    context = event.GetContext() # return a string representing the Context
    frame = event.GetFrame() # return the frame as an integer
    return event, label, context, frame

# But : trouver l'évènement le plus proche d'une position, frame donnée
# In : des données "data", l'ensemble des évènements (AllEvents), le label et le context recherché , et la position depuis laquel on recherche
# Out : l'évènement, et la frame cprrespondante
def closestEvent(data, AllEvents, label=0, context=0, start=1):
    if (label == 0) and (context == 0):
        return AllEvents.GetItem(0), AllEvents.GetItem(0).GetFrame()

    eventVIP = []   # Array qui contiendra tous les évènements correspondant au même label et même contexte que demandé
    numberEvent = AllEvents.GetItemNumber()     # Nombre d'évènements au total
    for num in range(numberEvent):              # On regarde tout les évènement
        event = AllEvents.GetItem(num)          # On récupère un évènement
        if (event.GetContext() == context) and (event.GetLabel() == label): # Test si on a les mêmes context et label
            eventVIP.append(event)              # On rajoute l'évènement

    if len(eventVIP) == 0:                      # Si on a trouvé aucun évènement recherché, on arrête
        return 0, 0

    dist = 1000                                 # On initialise une distance très grande, qui diminuera
    even = eventVIP[0]                          # On commence par le premier évènement
    for event in eventVIP:                      # On parcours les évènements
        if np.abs(event.GetFrame() - start) < dist:     # On test si la distance entre la position de départ et un évènement correspondant
            dist = np.abs(event.GetFrame() - start)     # On mémorise la nouvel distance
            even = event                                # On mémorise le nouvel évènement

    return even, even.GetFrame()

# But : trouver l'extremum le plus proche d'une position de départ
# In : position de départ "start", les indices (position x) d'extremum (les min ou les max)
# Out : position x de l'extremum, la distance par rapport au point de départ (start), et l'indice dans l'array des min ou max
def closestExtrem(start, indExtrem):  # Renvoie la position de l'extrem par rapport à la frame Start
    AllDistance = indExtrem - start         # Soustraction d'un vecteur par un scalaire, ici les positions des indices moins la position de départ (start)
    absDist = np.abs(AllDistance)           # On met en valeur absolue
    indexMinimDist = np.argmin(absDist)     # On récupère l'indice de la distance minimale
    positionExtrem = indExtrem[indexMinimDist]  # On récupère la position x de l'extremum
    distance = AllDistance[indexMinimDist]      # On récupère la distance (sans la valeur absolue)
    return positionExtrem, distance, indexMinimDist

# But : ploter
def plotPosi(acq, position, axe, ax, event=0):

    dicoAxe = {"x" : 0, "y" : 1, "z" : 2}
    data = np.array(acq.GetPoint(position).GetValues()[:, dicoAxe[axe]])
    n_frames, first_frame, last_frame = frameData(acq)
    Min, Max = minLocal(data), maxLocal(data)
    pas = selectStep(acq, position)



    # Plot part
    ax.plot(np.array(range(first_frame-1, last_frame)), data, 'k')
    ax.plot(Min, data[Min], 'o b')
    ax.plot(Max, data[Max], 'o', color='purple')
    ax.axvline(x = pas[0], color = 'cyan')
    ax.axvline(x = pas[1], color = 'cyan')
    ax = plotEvent(acq, ax)

    if (event != 0):
        print('Position de depart :', event.GetFrame())
        positionExtrem, distance, indexMinimDist = closestExtrem(event.GetFrame(), Max)
        ax.plot(positionExtrem, data[positionExtrem], 'o g')
        print('Position :', positionExtrem)
    plt.title(" Position = {} - axis = {}".format(position, axe))
    # ax.show(block = False)
    return ax

# But : modifier le bouton, pour ploter un seul graphe
# In : l'ensemble des fichiers ('files'), le sensor ('posiCombo'), l'axe ('axeCombo'), le bouton qui fait les plot ('buttonCombo'), et le numéro du fichier ('fileCombo')
# Pas de sortie, mais modifie l'action de bouton
def simple(files, posiCombo, axeCombo , buttonCombo, fileCombo):
    posiCombo['values'] = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'STRN', 'CLAV', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'RTHI', 'LKNE', 'RKNE', 'LTIB', 'RTIB', 'LANK', 'RANK', 'LHEE', 'RHEE', 'RTOE', 'LTOE']
    posiCombo.current(0)
    buttonCombo["text"] = "PLOT"
    buttonCombo["command"] = lambda: onePlot(files, posiCombo, axeCombo, fileCombo )

# But : modifier le bouton, pour ploter deux graphes
# In : l'ensemble des fichiers ('files'), le sensor ('posiCombo'), l'axe ('axeCombo'), le bouton qui fait les plot ('buttonCombo'), et le numéro du fichier ('fileCombo')
# Pas de sortie, mais modifie l'action de bouton
def double(files, posiCombo, axeCombo , buttonCombo, fileCombo):
    posiCombo['values'] = ["FHD", "BHD", "SHO", "ELB", "WRA", "WRB", "ASI", "PSI", "THI", "KNE", "TIB", "ANK", "HEE", "TOE"]
    posiCombo.current(0)
    buttonCombo["text"] = "PLOT x2"
    buttonCombo["command"] = lambda: twoPlot(files, posiCombo, axeCombo, fileCombo )

# But : ploter un seul graphe
# In : tous les fichiers ('files'), le capteur ('posiCombo'), l'axe ('axeCombo'), et le numéro du fichier ('fileCombo')
def onePlot (files, posiCombo, axeCombo, fileCombo ):
    acq = files[int(fileCombo.get())]             # voir le chapitre sur les événements
    n_frames, first_frame, last_frame = frameData(acq)
    plt.figure(figsize=(9,7))
    guiPlot = plt.subplot()
    guiPlot = plotPosi(acq, posiCombo.get(), axeCombo.get(), guiPlot)
    plt.show(block=False)

# But : ploter un deux graphe s
# In : tous les fichiers ('files'), le capteur ('posiCombo'), l'axe ('axeCombo'), et le numéro du fichier ('fileCombo')
def twoPlot(files, posiCombo, axeCombo, fileCombo ):               # voir le chapitre sur les événements
    acq = files[int(fileCombo.get())]
    n_frames, first_frame, last_frame = frameData(acq)
    dr = 'R' + posiCombo.get()
    ga = 'L' + posiCombo.get()
    plt.figure(figsize=(9,7))
    guiPlot = plt.subplot(2,1,1)
    guiPlot = plotPosi(acq, dr, axeCombo.get(), guiPlot)
    guiPlot = plt.subplot(2,1,2)
    guiPlot = plotPosi(acq, ga, axeCombo.get(), guiPlot)
    plt.show(block=False)

# But : créer une fenêtre GUI pour ploter les graphes facilement
# In : l'ensemble des fichiers
def GUIplot(files):
    acq = files[0]
    metadata = acq.GetMetaData()
    point_labels = list(metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString())

    win = Tk()
    win.title("BTK Project")
    # win.geometry("500x100")

    ttk.Label(win, text="Choix du capteur").grid(column=1, row=0)
    posiCombo = ttk.Combobox(win, values=point_labels)
    posiCombo.grid(column=1, row=1)
    ttk.Label(win, text="Choix de l'axe").grid(column=2, row=0)
    axeCombo = ttk.Combobox(win, values=["x", "y", "z"])
    axeCombo.grid(column=2, row=1)
    ttk.Label(win, text="Choix du fichier").grid(column=0, row=0)
    fileCombo = ttk.Combobox(win, values=list(range(len(files))))
    fileCombo.grid(column=0, row=1)
    posiCombo.current(newindex=28)
    axeCombo.current(2)
    fileCombo.current(0)


    buttonCombo = Button (win, text="PLOT", command= lambda: onePlot(files, posiCombo, axeCombo, fileCombo ))
    buttonCombo.grid(column=3, row=1)



    v = IntVar()
#    v.set(1)
    R1 = Radiobutton(win, text="Plot unique", variable=v, value=1, command= lambda: simple(files, posiCombo, axeCombo , buttonCombo, fileCombo))
    R1.grid(column=0, row=2)
    R2 = Radiobutton(win, text="Double Plot", variable=v, value=2, command= lambda: double(files, posiCombo, axeCombo , buttonCombo, fileCombo))
    R2.grid(column=1, row=2)
    v.set(1)

    win.mainloop()

# But : prédire la distance moyenne entre un event et un extremum
# In : tous les fihiers de training ('trainingSet'), 'eventLabel' et 'eventContext' servent à trouver l'event dans les données du sensor 'capteur',
# 'genre' sert à savoir si on veut la distance avec un min ou un max
def predictDist(trainingSet, eventLabel, eventContext, capteur, genre, taux):

    arrayDist = []      # Initialisation du vecteur contenant les distances, de taille len(trainingSet)

    for acr in trainingSet:     # On calcule la distance à travers tous les training files
        data = np.array(acr.GetPoint(capteur).GetValues()[:, 2])        # On récupère les données
        Min, Max = minLocal(data), maxLocal(data)                       # On récupère les extremaux
        event, eventFrame = closestEvent(data, acr.GetEvents(), eventLabel, eventContext)   # On récupère la position de l'event
        if genre == 'min':      # Si on veut faire avec min
            positionExtrem, distance, indexMinimDist = closestExtrem(eventFrame, Min)
        if genre == 'max':
            positionExtrem, distance, indexMinimDist = closestExtrem(eventFrame, Max)
        arrayDist.append(-distance)         # On mais le signe - car grace à ça, ça devient la distance de l'extremum vers l'event (et non plus l'inverse)
    return np.mean(arrayDist)               # On renvoie la moyenne

# But : trouver à quelles frames commencent et finissent les pas
# In : les données ('acq'), l'ensemble des capteurs ('capteurSet')
# Out : array contenant les frames des débuts et fins de pas
def pasR(acq, capteurSet):
    if capteurSet[0][0] == 'L':     # On regarde si le premier capteur concernent le pied gauche
        capteur = 'LANK'
    else:
        capteur = 'RANK'
    taux = 0.5      # le taux, pour fixer le seuil, afin d'éviter le min et max qui se chavauchent
    data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])    # On récupère les données
    indMax = np.ravel(maxLocal(data))                           # On récupère les max, qui représentent les débuts ou fins des pas
    seuil = np.mean(np.max(data)*taux + np.min(data)*(1-taux))  # On calcule le seuil
    pasRank = indMax[data[indMax] > seuil]                      # On prend que ce qui dépasse du seuil


    pasPlus = np.append(0, pasRank)                             # On rajoute la premiere et derniere frame
    pasPlus = np.append(pasPlus, len(data)-1)

    return pasPlus, pasRank

# But : prédire la position des pas / events
# In : les données ('acr'), la distance moyenne entre event et extremum ('moyenne'), tous les capteurs qu'on souhaite utilisé ('capteurSet'), et si c'est un min ou max ('genre')
def calcPredic(acr, moyenne, capteurSet, genre):
    predi = {}      # Dico qui contiendra toutes les prédictions, de même taille que capteurSet
    taux = 0.8      # Taux pour le seuil

    pasPlus, pasRank = pasR(acr, capteurSet)        # On récupère les frames des pas
    dataTout = np.array(acr.GetPoint(capteurSet[0]).GetValues()[:, 2])             # La totalité des données

    for capteur in capteurSet:      # Initialisation du dico predi, qui contiendra des arrays de prédictions des events
        predi[capteur] = []

    pasRank = pasPlus               # On prend les bouts (première et dernière frame)

    for i in range(len(pasRank)-1):         # On parcourt les pas

        for capteur in capteurSet:          # Chaque capteur fait sa prédiction
            data = np.array(acr.GetPoint(capteur).GetValues()[pasRank[i]:pasRank[i+1], 2])
            seuil = np.mean(np.max(data)*taux + np.min(data)*(1-taux))
            Min, Max = minLocal(data), maxLocal(data, 1)        # Extremum, mais juste dans le pas

            if genre[capteur] == 'max':
                Max = Max[data[Max] > seuil]
                if len(Max) == 0:       # Au cas où...
                    Max = [len(dataTout)-1]
                predi[capteur].append(round(pasRank[i] + Max[-1] + moyenne[capteur]))   # + pasRank, à cause du décalage causé par le pas
            if genre[capteur] == 'min':
                Min = Min[data[Min] < seuil]
                if len(Min) == 0:
                    Min = [len(dataTout)-1]
                predi[capteur].append(round(pasRank[i] + Min[-1] + moyenne[capteur]))

    return predi

# But : transformer un dico en array, ayant le même ordre que capteurSet
# In : le dictionnaire à convertir, et l'ensemble des capteurs pour avoir un ordre
# Out : un array dont chaque élément était dans le dico
def dic2mat(dico, capteurSet):
    dictArray = []
    for capteur in capteurSet:
        dictArray.append(dico[capteur])
    return dictArray

# But : Initialisation des poids
# In : L'ensemble des capteurs, pour associé à chaque capteur un poids
# Out : les poids, tous égaux
def weightCreation(capteurSet):
    weight = {}
    for capteur in capteurSet:
        weight[capteur] = 1/len(capteurSet)     # Tous les poids sont égaux au départ

    return weight

# But : Mettre à jour les poids, selon les résultats des prédictions
# In : les prédictions dans un array, ayant le même ordre que capteurSet ('arrayPredi'), la frame exacte de l'event qui servira à comparer avec la prédiction
# les derniers poids, utilisés pas la prédiction ('weight'), capteurSet, et 'nbTest' correspond au nombre de test fait jusqu'à maintenant
# Out : les poids mis à jour
def weightUpDate(arrayPredi, evenFrame, weight, capteurSet, nbTest):

    # Initialisation des variables
    nouvScore = {}      # Score selon la réussite de la prédiction
    nouvWeight = {}     # Nouveaux poids, mis à jour
    nbWeight = len(weight)  # Nombre de poids
    somme = np.sum(np.array(range(len(weight))))        # Pour la normalisation

    ecart = np.abs(arrayPredi-evenFrame)                 # Calcul de l'écart entre la frame exact et les prédictions
    indexMin = np.argmin(ecart, axis=1)                  # On récupère l'index de l'écart le plus bon (i.e. la bonne prédiction)
    classement = np.argsort(ecart[:, indexMin[0]])       # On fait un classement des meilleurs capteurs
    capteurSet = np.asarray(capteurSet)

    n = 0       # n représente les points, des jetons pour signifie la bonne réussite de la prédiction
    for capteur in capteurSet[classement]:
        n += 1
        score = nbWeight - n            # Pour trier de façon décroissante
        nouvScore[capteur] = score/somme    # On normalise, pour la somme donne 1

    # Mise à jour des poids
    for capteur in capteurSet:
        nouvWeight[capteur] = (weight[capteur]*nbTest + nouvScore[capteur])/(nbTest+1)

    return nouvWeight


# But : On effectue les tests sur l'ensemble des fichiers de tests
# In : les fichiers tests, etc... (voir predictDist)
# Out : l'écart de frame entre la prédiction et la réalité
def test(testingSet, eventLabel, eventContext, moyenne, weight, capteurSet, genre):
    erreur = []
    for acr in testingSet:
        even, evenFrame = closestEvent(acr, acr.GetEvents(), eventLabel, eventContext)
        predi = calcPredic(acr, moyenne, capteurSet, genre)

        arrayPredi, arrayWeight = np.asarray(dic2mat(predi, capteurSet)) , dic2mat(weight, capteurSet)        # Transformation en array
        meanPrediction = np.round(np.dot(arrayWeight, arrayPredi))         # On applique les poids pour faire la moyenne pondérée
        erreur.append(np.min(np.abs(meanPrediction - evenFrame)))   # On trouve l'erreur (qui est la distance entre l'event et notre prediction correspondante)

    return erreur