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
from functionFile import *

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
