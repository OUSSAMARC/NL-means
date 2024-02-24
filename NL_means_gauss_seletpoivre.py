import numpy as np
from PIL import Image
import os


script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir)) 
image_path = os.path.join(parent_directory, 'Images et vidéos', 'Cristiano.jpg') 

image_pil = Image.open(image_path).convert('L')  # convert('L') pour charger l'image en niveaux de gris

# Convertir l'image en tableau NumPy et normaliser les valeurs des pixels
image = np.array(image_pil) / 255.0
noisex = np.random.normal(0, 0.1, image.shape).astype(np.float64)
noisy = image + noisex

# Définition du bruit sel et poivre
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Salt noise
    salt_pixels = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_pixels] = 1

    # Pepper noise
    pepper_pixels = np.random.rand(*image.shape) > 1-pepper_prob
    noisy_image[pepper_pixels] = 0

    return noisy_image
salt_probability = 0.02 # Probabilité de bruit sel
pepper_probability = 0.02 # Probabilité de bruit poivre

noisy_image = add_salt_and_pepper_noise(noisy, salt_probability, pepper_probability)

"""
Pour la recherche des voisins les plus proches du pixel bruité, on utilise deux fonctions,
chacune dépendant du type de bruit affectant le pixel.
"""
def extract_closest_neighbors_selpoivre(x_i, y_i, image, half_window, num_neighbors, t):
  
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
            distance_squared = np.sum(ker_selpoivre * (N_i - N_j) ** 2)
            distances.append(((j_y, j_x), distance_squared))

    # Trier les distances et extraire les coordonnées des voisins les plus proches
    distances.sort(key=lambda x: x[1])
    neighbors = [coord for coord, _ in distances[:num_neighbors]]

    return neighbors


def extract_closest_neighbors_gaussien(x_i, y_i, image, half_window, num_neighbors ):
 
    t=5

    height, width = image.shape[:2]
    

    # Définir les limites de la fenêtre centrée sur (x_i, y_i)
    start_x, end_x =  x_i - half_window,  x_i + half_window + 1
    start_y, end_y =  y_i - half_window,  y_i + half_window + 1
    
    start_X, end_X =  max(x_i - t,0) ,  min(x_i+t+1,image.shape[1]-half_window)
    start_Y, end_Y = max( y_i - t,0),  min(y_i+t+1,image.shape[0]-half_window)
    

    # Extraire la fenêtre Ni
    N_i = image[start_y:end_y, start_x:end_x]
    distances=[]

    

    for j_x in range(max(start_X,half_window),end_X):
        for j_y in range(max(start_Y,half_window),end_Y):
            #if (x_i, y_i) != (j_x, j_y):  # Exclure le cas où (x_i, y_i) est égal à (j_x, j_y)
                # Définir les limites de la fenêtre Nj
                
                
                start_j_x, end_j_x =  j_x - half_window,  j_x + half_window +1
                start_j_y, end_j_y =  j_y - half_window,  j_y + half_window +1
                
                # Extraire la fenêtre Nj
                N_j = image[start_j_y:end_j_y, start_j_x:end_j_x]

                # Vérifier les dimensions et redimensionner si nécessaire
                

                # Calculer la distance euclidienne
                distance_squared = np.sum(kernel*(N_i - N_j) ** 2)
                distances.append(((j_y, j_x), distance_squared))


    

    # Sort the distances and get the coordinates of the closest neighbors
    distances.sort(key=lambda x: x[1])
    neighbors = [coord for coord, _ in distances[:num_neighbors]]

    return neighbors

# Neighborhood window size
f = 2
# Similarity window size
t = 5

# Making gaussian kernel
su = 1
sm = 0
ks = 2 * f + 1
# Création du Kernel adapté au sel et poivre 
ker_selpoivre = np.zeros((ks, ks))

for x in range(ks):
        for y in range(ks):
            if x ==f and y==f :
                ker_selpoivre[x, y]=0
            else :
               ker_selpoivre[x, y] = (-1)/2
sm = sm +ker_selpoivre[x, y]
kernel_selpoivre = ker_selpoivre / f
kernel_selpoivre = kernel_selpoivre / sm

# Création du Kernel adapté au bruit gaussien
ker = np.zeros((ks, ks))

for x in range(ks):
        for y in range(ks):
            ab = x - f
            cd = y - f
            ker[x, y] = 100 * np.exp(((ab * ab) + (cd * cd)) / (-2 * (su * su)))
            sm = sm + ker[x, y]

kernel = ker / f
kernel = kernel / sm

def apply_algorithm_to_channel(image):
    
    # Assign a clear output image
    cleared = np.zeros_like(image)

    # Degree of filtering
    h = 0.1
    
    noisy2 = np.pad(image, ((f, f), (f, f)), mode='symmetric')
    
    # On parcourt les pixels et on vérifie s'il s'agit d'un bruit sel et poivre ou bruit gaussien
    for i in range(f, noisy2.shape[0] -f):
        for j in range(f, noisy2.shape[1]-f ):
          
            if noisy2[i,j] == 1 or noisy2[i,j] == 0 :
            # Neighborhood of concerned pixel (similarity window)
                    W1 = noisy2[i - f:i + f + 1, j - f:j + f + 1]
                    
                    NL = 0
                    Z = 0
                    L=extract_closest_neighbors_selpoivre(j,i, noisy2 , f, 25,t)
                    
                    for r, s in L:
                            W2 = noisy2[r - f:r + f + 1, s - f:s + f + 1]
                            d2 = np.sum(kernel_selpoivre * (W1 - W2) * (W1 - W2))
                            sij = np.exp(-d2 / (h * h))
                            
                            if r == i and s==j :
                                sij=0
                            
                            Z = Z + sij
                            NL = NL + (sij * noisy2[r,s])
            
                        
                    cleared[i - f, j - f] = NL / Z
            else :
                W1 = noisy2[i - f:i + f + 1, j - f:j + f + 1]
               
                
                NL = 0
                Z = 0
                L=extract_closest_neighbors_gaussien(j,i, noisy2 , f, 25)
               
                for r, s in L:
                        
                        W2 = noisy2[r - f:r + f + 1, s - f:s + f + 1]
                        d2 = np.sum(kernel * (W1 - W2) * (W1 - W2))
                        sij = np.exp(-d2 / (h * h))
                        Z = Z + sij
                        NL = NL + (sij * noisy2[r, s])
                
                cleared[i - f, j - f] = NL / Z
                    
    return cleared



