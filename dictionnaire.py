# In[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os           # Pour lire le nom des fichiers
from array import *



def dicoCapteurGenre(type):

    if type == '\ITW':
        capteurSet_Off_Right = ['RANK', 'RHEE', 'C7', 'LHEE', 'RPSI', 'RKNE', 'T10', 'LFHD', 'LPSI']
        capteurSet_Off_Left = ['LANK', 'LHEE', 'RANK', 'C7', 'RHEE', 'LBHD', 'LSHO', 'LKNE']
        capteurSet_Strike_Right = ['LANK', 'LHEE', 'RANK', 'RHEE', 'LBHD', 'LSHO', 'LPSI', 'LKNE']
        capteurSet_Strike_Left = ['RANK', 'RHEE', 'LANK', 'C7', 'LHEE', 'RFHD', 'RBHD', 'RSHO', 'RPSI', 'RKNE', 'CLAV']

    if type == '\CP':
        capteurSet_Off_Right = ['RANK', 'RHEE', 'C7', 'LHEE', 'CLAV', 'RKNE', 'T10']
        capteurSet_Off_Left = ['LANK', 'LHEE', 'T10', 'LKNE']
        capteurSet_Strike_Right = ['LANK', 'LHEE', 'C7', 'T10', 'RANK', 'RHEE', 'LKNE']
        capteurSet_Strike_Left = ['RHEE', 'RANK', 'C7', 'RKNE', 'CLAV']

    if type == '\FD':
        capteurSet_Off_Right = ['RANK', 'RHEE', 'C7', 'RKNE']
        capteurSet_Off_Left = ['LANK', 'LHEE', 'RHEE', 'LSHO', 'LKNE']
        capteurSet_Strike_Right = ['LANK', 'LHEE', 'RANK', 'CLAV', 'T10']
        capteurSet_Strike_Left = ['RANK', 'RHEE', 'C7', 'RFHD', 'RBHD', 'RPSI', 'RKNE', 'CLAV']


    genre_Off_Right = {'RANK': 'max', 'RHEE': 'max', 'C7': 'min', 'RTIB': 'max', 'RKNE': 'min',
                       'LANK': 'min', 'LHEE': 'min', 'RFHD': 'min', 'LFHD': 'min', 'RBHD': 'min', 'RELB': 'min',
                       'RSHO': 'min', 'RPSI': 'min', 'RKNE': 'min', 'CLAV': 'min', 'LPSI': 'min', 'T10' : 'min'  }

    genre_Off_Left = {'LANK': 'max', 'LHEE': 'max', 'C7': 'min', 'LTIB': 'max', 'LKNE': 'min',
                      'RANK': 'max', 'RHEE': 'min', 'LFHD': 'min', 'RFHD': 'min', 'LBHD': 'min', 'LELB': 'max',
                      'LSHO': 'min', 'LPSI': 'min', 'LKNE': 'min', 'CLAV': 'min', 'T10' : 'min'}

    genre_Strike_Right = {'LANK': 'max', 'LHEE': 'max', 'C7': 'min', 'LTIB': 'max', 'LKNE' : 'min',
                          'RANK': 'min', 'RHEE': 'min', 'LFHD': 'min', 'RFHD': 'min', 'LBHD': 'min', 'LELB': 'max',
                          'LSHO': 'min', 'LPSI': 'min', 'LKNE': 'min', 'CLAV': 'min', 'T10' : 'min'}

    genre_Strike_Left = {'RANK': 'max', 'RHEE': 'max', 'C7': 'min', 'RTIB': 'max', 'RKNE': 'min',
                         'LANK': 'min', 'LHEE': 'min', 'RFHD': 'min', 'LFHD': 'min', 'RBHD': 'min', 'RELB': 'max',
                         'RSHO': 'min', 'RPSI': 'min', 'RKNE': 'min', 'CLAV': 'min'}


    # Left   Right
    # Strike  Off
    capteurDico = {}
    capteurDico['Foot_Off_GS'] = {}
    capteurDico['Foot_Strike_GS'] = {}
    capteurDico['Foot_Off_GS'] = {'Left': capteurSet_Off_Left, 'Right': capteurSet_Off_Right}
    capteurDico['Foot_Strike_GS'] = {'Left': capteurSet_Strike_Left, 'Right': capteurSet_Strike_Right}

    genreDico = {}
    genreDico['Foot_Off_GS'] = {}
    genreDico['Foot_Strike_GS'] = {}
    genreDico['Foot_Off_GS'] = {'Left': genre_Off_Left, 'Right': genre_Off_Right}
    genreDico['Foot_Strike_GS'] = {'Left': genre_Strike_Left, 'Right': genre_Strike_Right}

    return capteurDico, genreDico
