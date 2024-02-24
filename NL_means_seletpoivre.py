import numpy as np
from PIL import Image
import os 

# Chemin du répertoire du script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Répertoire parent
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir)) 
# Chemin de l'image
image_path = os.path.join(parent_directory, 'Images et vidéos', 'Cristiano.jpg') 

# Charger l'image en niveaux de gris avec PIL
image_pil = Image.open(image_path).convert('L')  # convert('L') pour charger l'image en niveaux de gris

# Convertir l'image en tableau NumPy et normaliser les valeurs des pixels
image = np.array(image_pil) / 255.0


def extract_closest_neighbors(x_i, y_i, image, half_window, num_neighbors, t):
  
    height, width = image.shape[:2]
    # Définir les limites du patch centré sur (x_i, y_i)
    start_x, end_x = x_i - half_window, x_i + half_window + 1
    start_y, end_y = y_i - half_window, y_i + half_window + 1
    # Définir la grande fenêtre
    start_X, end_X = max(x_i - t, 0), min(x_i + t + 1, image.shape[1] - half_window)
    start_Y, end_Y = max(y_i - t, 0), min(y_i + t + 1, image.shape[0] - half_window)

    # Extraire la fenêtre Ni
    N_i = image[start_y:end_y, start_x:end_x]
    # Une liste où l'on va stocker les distances entre i et chaque pixel j de la grande fenêtre
    distances = []

    for j_x in range(max(start_X, half_window), end_X):
        for j_y in range(max(start_Y, half_window), end_Y):
            # Définir les limites de la fenêtre Nj
            start_j_x, end_j_x = j_x - half_window, j_x + half_window + 1
            start_j_y, end_j_y = j_y - half_window, j_y + half_window + 1

            # Extraire la fenêtre Nj
            N_j = image[start_j_y:end_j_y, start_j_x:end_j_x]

            # Calculer la distance euclidienne
            distance_squared = np.sum(kernel * (N_i - N_j) ** 2)
            distances.append(((j_y, j_x), distance_squared))

    # Trier les distances et extraire les coordonnées des voisins les plus proches
    distances.sort(key=lambda x: x[1])
    neighbors = [coord for coord, _ in distances[:num_neighbors]]

    return neighbors
"""
Le kernel qu'on utilise cette fois ci est seuelement adapté au sel et poivre 
car il ne prend pas en considération le pixel centrale (bruité) qui est totalement
endommagé et qui ne contient donc pas d'information utile. Le pixel sel ou poivre 
ne participe donc pas dans le procecessus de la recherche des voisinage ppv dans la fonction extract_closest_neighbors
"""

# Taille de la fenêtre de voisinage
f = 2
# Taille de la fenêtre de similarité
t = 5
# Création du noyau gaussien
su = 1
sm = 0
ks = 2 * f + 1
ker = np.zeros((ks, ks))

for x in range(ks):
   for y in range(ks):
       if x == f and y == f:
               ker[x, y] = 0
       else:
              ker[x, y] = (-1) / 2
       sm = sm + ker[x, y]

kernel = ker / f
kernel = kernel / sm

def apply_algorithm_to_channel(image):

    # Assigner une image de sortie claire
    cleared = np.zeros_like(image)
    
    # Degré de filtrage
    h = 1
    
    noisy2 = np.pad(image, ((f, f), (f, f)), mode='symmetric')
  
    # Calcul de la sortie pour chaque pixel
    for i in range(f, noisy2.shape[0] - f):
        for j in range(f, noisy2.shape[1] - f):

            if noisy2[i, j] == 1 or noisy2[i, j] == 0:
                # Fenêtre de similarité du pixel concerné
                W1 = noisy2[i - f:i + f + 1, j - f:j + f + 1]
                NL = 0
                Z = 0
                L = extract_closest_neighbors(j, i, noisy2, f, 25, t)  # Voisins les plus proches
                
                for r, s in L:
                    W2 = noisy2[r - f:r + f + 1, s - f:s + f + 1]
                    d2 = np.sum(kernel * (W1 - W2) * (W1 - W2))
                    sij = np.exp(-d2 / (h * h))
                    """
                     Cette ligne attribue au poids w(i,i) poids du pixel bruité lui-même la valeur 0
                     le pixel sel ou poivre ne participe pas à la restauration de sa valeur
                    """
                    if r == i and s == j:
                        sij = 0
                    
                    Z = Z + sij
                    NL = NL + (sij * noisy2[r, s])
                cleared[i - f, j - f] = NL / Z
            else:
                cleared[i - f, j - f] = noisy2[i, j]
                    
    return cleared


# Définition du bruit sel et poivre
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Bruit sel
    salt_pixels = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_pixels] = 1

    # Bruit poivre
    pepper_pixels = np.random.rand(*image.shape) > 1 - pepper_prob
    noisy_image[pepper_pixels] = 0

    return noisy_image


# Paramètres de bruit (ajustez selon vos préférences)
salt_probability = 0.02  # Probabilité de bruit sel
pepper_probability = 0.02  # Probabilité de bruit poivre

# Ajouter le bruit sel et poivre
noisy_image = add_salt_and_pepper_noise(image, salt_probability, pepper_probability)

# Image débruitée finale
cleared_img = apply_algorithm_to_channel(noisy_image)



