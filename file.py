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

# But : trouvé tous les maximum locaux
# In : un vecteur de taille nx1
# Out : les positions x des max locaux (pas leur valeur y)
def maxLocal(a):
    TFarray = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True])   # Rempli un vecteur avec que des False, sauf lorsqu'une donnée dans le vecteur est plus grande que son voisin de droite et de gauche (il met alors True)
    indMax = np.where( TFarray == True )    # On récupère les index où il y a les True
    return np.ravel(indMax)

# Fonctions en cours, pas encore effective
def semiMaxLocal(a):
    TFarray = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] == a[1:], True])
    indSemiMax = np.where( TFarray == True )
    return indSemiMax

def findMinMin(data, Min):
    minMin = minLocal(data[Min])
    return Min[minMin]

# Pareil que maxLocal, mais pour trouver les minimum locaux
def minLocal(a):
    TFarray = np.array(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
    indMin = np.where( TFarray == True )
    return np.ravel(indMin)

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
    ax.legend((leftLineOff, rightLineStrike, rightLineOff), ('Left - Off', 'Right - Strike', 'Right - Off'))

    return plt

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
                files.append(os.path.join(r, file))
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
# In : des données "data", l'ensemble des évènements (AllEvents), le labelet le context recherché , et la position depuis laquel on recherche
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

def plotPosi(acq, first_frame, last_frame, position, axe, ax):
    import matplotlib.pyplot as plt

    dicoAxe = {"x" : 0, "y" : 1, "z" : 2}
    data = np.array(acq.GetPoint(position).GetValues()[:, dicoAxe[axe]])
    Min, Max = minLocal(data), maxLocal(data)

    # Plot part
    # ax = plt.subplot()
    ax.plot(np.array(range(first_frame, last_frame + 1)), data, 'k')
    ax.plot(Min, data[Min], 'o b')
    ax.plot(Max, data[Max], 'o', color='purple')
    # ax.plot(MinR[indexMinimDist], dataR[MinR[indexMinimDist]], 'o r')
    # ax.plot(minMinR, dataR[minMinR], 'o', color='orange')
    # ax.plot(autre, dataR[autre], 'o', color='brown')
    ax = plotEvent(acq, ax)
    ax.title(" Position = {} - axis = {}".format(position, axe))
    # ax.show(block = False)
    return ax

def simple(acq, first_frame, last_frame, posiCombo, axeCombo , buttonCombo):
    posiCombo['values'] = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'STRN', 'CLAV', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'RTHI', 'LKNE', 'RKNE', 'LTIB', 'RTIB', 'LANK', 'RANK', 'LHEE', 'RHEE', 'RTOE', 'LTOE']
    buttonCombo["text"] = "PLOT"
    buttonCombo["command"] = lambda: onePlot(acq, first_frame, last_frame, posiCombo, axeCombo )

def double(acq, first_frame, last_frame, posiCombo, axeCombo , buttonCombo):
    posiCombo['values'] = ["FHD", "BHD", "SHO", "ELB", "WRA", "WRB", "ASI", "PSI", "THI", "KNE", "TIB", "ANK", "HEE", "TOE"]
    buttonCombo["text"] = "PLOT x2"
    buttonCombo["command"] = lambda: twoPlot(acq, first_frame, last_frame, posiCombo, axeCombo )

def onePlot (acq, first_frame, last_frame, o, p ) :                     # voir le chapitre sur les événements
    figure2 = plt.figure(figsize=(9,7))
    guiPlot = plt.subplot()
    guiPlot = plotPosi(acq, first_frame, last_frame, o.get(), p.get(), guiPlot)
    guiPlot.show(block=False)

def twoPlot (acq, first_frame, last_frame, o, p ) :                     # voir le chapitre sur les événements
    dr = 'R' + o.get()
    ga = 'L' + o.get()
    figure2 = plt.figure(figsize=(9,7))
    guiPlot = plt.subplot(2,1,1)
    guiPlot = plotPosi(acq, first_frame, last_frame, dr, p.get(), guiPlot)
    guiPlot = plt.subplot(2,1,2)
    guiPlot = plotPosi(acq, first_frame, last_frame, ga, p.get(), guiPlot)
    guiPlot.show(block=False)

def GUIplot(acq, point_labels):
    # Infos utiles
    n_frames, first_frame, last_frame = frameData(acq)

    win = Tk()
    win.title("BTK Project")
    # win.geometry("500x100")

    ttk.Label(win, text="Choix du capteur").grid(column=0, row=0)
    posiCombo = ttk.Combobox(win, values=list(point_labels))
    posiCombo.grid(column=0, row=1)
    ttk.Label(win, text="Choix de l'axe").grid(column=1, row=0)
    axeCombo = ttk.Combobox(win, values=["x", "y", "z"])
    axeCombo.grid(column=1, row=1)


    buttonCombo = Button (win, text="PLOT", command= lambda: onePlot(acq, first_frame, last_frame, posiCombo, axeCombo ))
    buttonCombo.grid(column=2, row=1)



    v = IntVar()
    v.set(1)
    R1 = Radiobutton(win, text="Plot unique", variable=v, value=1, command= lambda: simple(acq, first_frame, last_frame, posiCombo, axeCombo , buttonCombo))
    R1.grid(column=0, row=2)
    R2 = Radiobutton(win, text="Double Plot", variable=v, value=2, command= lambda: double(acq, first_frame, last_frame, posiCombo, axeCombo , buttonCombo))
    R2.grid(column=1, row=2)

    win.mainloop()
