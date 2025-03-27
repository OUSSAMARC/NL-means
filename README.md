# Projet Traitement d'Images NL Means 

## Description 

Ce projet utilise la méthode NL means afin de débruiter des images ou vidéos, en noir et blanc et en couleurs. Le projet traite également des optimisations de l'algorithme du NL means afin d'améliorer sa précision et son efficacité. Le projet traite deux types de bruit : le bruit gaussien et le bruit de sel et de poivre.

## Utilisation 

Dans le dossier code se trouve les script python suivant : 

- "projet_TI_couleur" : ce script python bruite d'abord l'image en couleur avec un bruit gaussien 
additif, puis la débruite avec la méthode NL Means. Le résultat sera enregistré 
dans le dossiers "Images et vidéos Résultats".

- "projet_TI_bruitage_video" : ce script python bruite la vidéo avec un bruit gaussien 
additif. La vidéos bruité se trouve ensuite dans Images et vidéos.

- "projet_TI_video" : ce script python permet de débruiter la vidéo préalablement 
bruiter grâce à la méthode NL Means et le résultat sera enregistré dans le dossier 
"Images et vidéos Résultats".

- "projet_TI_seletpoivre" ce script python permet d'ajouter un bruit sel et poivre à 
une image, puis la débruite avec la méthode NL Means adapté au sel et poivre.

- "projet_TI_gauss_seletpoivre" ce script pyhton permet d'ajouter un bruit sel et 
poivre et un bruit gaussien à une image, puis la débruite avec la méthode NL Means.


- "projet_TI_avec_optimisation_ppv" : ce script python bruite d'abord l'image en noir et blanc avec 
un bruit gaussien additif, puis la débruite avec la méthode NL Means optimisé avec la méthode ppv (plus proches voisinage). Le résultat sera enregistré dans le dossiers "Images et vidéos Résultats".

- "projet_TI_avec_optimisation_h" : ce script python bruite d'abord l'image en noir et blanc avec 
un bruit gaussien additif, puis la débruite avec la méthode NL Means. Le résultat 
sera enregistré dans le dossiers "Images et vidéos Résultats". Cependant ce script 
est une optimisation du script "projet_TI_avec_ppv" car on cherche le meilleur h 
(coefficient de filtrage) pour chaque pixel, qui permet d'avoir un moyennage optimal.


## Structure des dossiers du projet 

> 📂 Projet Traitement d'images
>> 📂 Code : où se trouve tous les codes utilisés
>> 📂 Bibliographie 
>> 📂 Images et vidéos : où se trouve les images et vidéos utilisées dans les codes 
>> 📂 Images et vidéos Résultats : où se trouve toutes les images et vidéos après NL means
