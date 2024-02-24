import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Obtention du répertoire du script et de son répertoire parent
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir))
image_path = os.path.join(parent_directory, 'Images et vidéos', 'oeil.png')

# Charger l'image en niveaux de gris avec PIL
image_pil = Image.open(image_path).convert('L')

# Convertir l'image en tableau NumPy et normaliser les valeurs des pixels
img = np.array(image_pil) / 255.0

# La fonction d'optimisation PPV réside dans cette fonction qui cherche les pixels num_neighbors ayant les voisinages les plus proches du voisinage du pixel i bruité
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

# Demi-taille du patch
f = 2
# Demi-taille de la grande fenêtre de recherche
t = 5

# Définition du noyau gaussien
su = 1
sm = 0
ks = 2 * f + 1
ker = np.zeros((ks, ks))

for x in range(ks):
    for y in range(ks):
        ab = x - f
        cd = y - f
        ker[x, y] = 10 * np.exp(((ab * ab) + (cd * cd)) / (-2 * (su * su)))
        sm = sm + ker[x, y]

kernel = ker / f
kernel = kernel / sm

# Ajout du bruit gaussien à l'image
noisex = np.random.normal(0, 0.05, img.shape).astype(np.float64)
noisy = img + noisex

# Assignation de "cleared" à l'image finale (débruitée)
cleared = np.zeros_like(img)

# Degré de filtrage
h = 0.07

# Remplissage des bords (padding)
noisy2 = np.pad(noisy, ((f, f), (f, f)), mode='symmetric')

# Maintenant, nous allons calculer la sortie de chaque pixel
for i in range(f, noisy2.shape[0] - f):
    for j in range(f, noisy2.shape[1] - f):
        # Voisinage du pixel concerné
        W1 = noisy2[i - f:i + f + 1, j - f:j + f + 1]
        NL = 0
        Z = 0
        # Nous extrayons les 25 pixels qui ont les voisinages les plus proches du pixel (i, j)
        L = extract_closest_neighbors(j, i, noisy2, f, 25, t)
        
        for r, s in L:
            # Voisinage du pixel (s, r)
            W2 = noisy2[r - f:r + f + 1, s - f:s + f + 1]
            # Calcul de la distance euclidienne
            d2 = np.sum(kernel * (W1 - W2) * (W1 - W2))
            sij = np.exp(-d2 / (h * h))
            Z = Z + sij
            NL = NL + (sij * noisy2[r, s])
        # Sortie du pixel
        cleared[i - f, j - f] = NL / Z






