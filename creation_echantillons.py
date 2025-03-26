# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 22:13:53 2025

@author: Elsa_Ehrhart & Guillaume Staub
"""

import os
import shutil
import random

def selection_annee_aleatoire(dossier):
    #liste des régions sans les dossiers cachés
    regions=[d for d in os.listdir(dossier) if (not d.startswith(".DS_Store") and not d.startswith("._.DS_Store"))]
    #region aléatoire
    region=random.choice(regions)
    region_path=os.path.join(dossier,region)
    annees=[d for d in os.listdir(region_path) if (not d.startswith(".DS_Store") and not d.startswith("._.DS_Store"))]
    annee=random.choice(annees)
    annee_path=os.path.join(region_path,annee)
    return region,annee,annee_path


dossier_apprentissage = ".\ech_apprentissage"
dossier_test = ".\ech_test"
data=".\data_for_climix"
taille_test=40


if os.path.isdir(dossier_apprentissage):  # Vérifie si le dossier existe
    shutil.rmtree(dossier_apprentissage)

if os.path.isdir(dossier_test):
    shutil.rmtree(dossier_test)
    
#os.mkdir(dossier_apprentissage)
os.mkdir(dossier_test)
shutil.copytree(data,dossier_apprentissage)

#on copie tout le dossier de data et on selectionne aléatoirement taille test années pour lesquelles on déplace les données dans le dossier de test    

for i in range (taille_test):
    
    region,annee,annee_path=selection_annee_aleatoire(dossier_apprentissage)
    #copie dans le repertoire de test
    destination=os.path.join(dossier_test,region)
    if not os.path.isdir(destination):  # Vérifie si le dossier existe
        os.mkdir(destination)

    shutil.move(annee_path,destination)
    

    
