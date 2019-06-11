from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from btk import btk

import matplotlib

import matplotlib.pyplot as plt

# But : trouvé tous les maximum locaux
# In : un vecteur de taille nx1
# Out : les positions x des max locaux (pas leur valeur y)
def maxLocal(a):
    TFarray = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True])   # Rempli un vecteur avec que des False, sauf lorsqu'une donnée dans le vecteur est plus grande que son voisin de droite et de gauche (il met alors True)
    indMax = np.where( TFarray == True )    # On récupère les index où il y a les True
    return indMax

# Fonctions en cours, pas encore effective
def semiMaxLocal(a):
    TFarray = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] == a[1:], True])
    indSemiMax = np.where( TFarray == True )
    return indSemiMax

# Pareil que maxLocal, mais pour trouver les minimum locaux
def minLocal(a):
    TFarray = np.array(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
    indMin = np.where( TFarray == True )
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
    ax.legend((leftLineOff, rightLineStrike, rightLineOff), ('Left - Off', 'Right - Strike', 'Right - Off'))

    return plt

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
def selectionnePas(acq, event):
    if event.GetContext() == 'Left':
        capteur = 'LHEE'
    else:
        capteur = 'RHEE'
    data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
    indMax = np.ravel(maxLocal(data[:]))
    event_frame = event.GetFrame()
    for i in range(len(indMax)):
        if indMax[i]>event_frame:
            start_step = indMax[i-1]
            end_step = indMax[i]
            break
    return start_step, end_step


# But : Récupérer les données
# In : path des données (Attention : le chemin commence de là où est le fichier)
# Out : les données
def initial(pathFile):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(pathFile)
    reader.Update()
    acq = reader.GetOutput()
    return acq

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
def closestEvent(data, AllEvents, label, context, start=1):
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
