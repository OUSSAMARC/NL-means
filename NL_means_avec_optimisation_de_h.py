import cv2
import numpy as np
import math
import os 


script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir)) 
image_path  = os.path.join(parent_directory, 'Images et vidéos', 'Cristiano.jpg') 

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0

def extract_closest_neighbors(x_i, y_i, image, half_window, num_neighbors , t):

    height, width = image.shape[:2]
    # Définir les limites du patch centré sur (x_i, y_i)
    start_x, end_x =  x_i - half_window,  x_i + half_window + 1
    start_y, end_y =  y_i - half_window,  y_i + half_window + 1
    # Définir la grande fenetre
    start_X, end_X =  max(x_i - t,0) ,  min(x_i+t+1,image.shape[1]-half_window)
    start_Y, end_Y = max( y_i - t,0),  min(y_i+t+1,image.shape[0]-half_window)
    

    # Extraire la fenêtre Ni
    N_i = image[start_y:end_y, start_x:end_x]
    # Une liste ou on va stocker les distances entre i et chaque pixel j de la grande fenetre
    distances=[]

    for j_x in range(max(start_X,half_window),end_X):
        for j_y in range(max(start_Y,half_window),end_Y):
                # Définir les limites de la fenêtre Nj
                
                
                start_j_x, end_j_x =  j_x - half_window,  j_x + half_window +1
                start_j_y, end_j_y =  j_y - half_window,  j_y + half_window +1
                
                # Extraire la fenêtre Nj
                N_j = image[start_j_y:end_j_y, start_j_x:end_j_x]
            
                # Calculer la distance euclidienne
                distance_squared = np.sum(kernel*(N_i - N_j) ** 2)
                distances.append(((j_y, j_x), distance_squared))


    

    # Trier les distances extraires les coordonnées des voisinage les plus proches
    distances.sort(key=lambda x: x[1])
    neighbors = [coord for coord, _ in distances[:num_neighbors]]

    return neighbors

#fonction qui calcule pour chaque pixel le coefficiant h appropié au coefficant 1/Z qu'on souhaite avoir
def calculate_h(x_i, y_i,L, image,wmeme,half_window) : #wmeme = 1/Z
    
    height, width = image.shape[:2]
    Nbr_pixels=len(L)

    # Définir les limites du patch centrée sur (x_i, y_i)
    start_x, end_x =  x_i - half_window,  x_i + half_window + 1
    start_y, end_y =  y_i - half_window,  y_i + half_window + 1

    # Extraire la fenêtre Ni
    N_i = image[start_y:end_y, start_x:end_x]

    total_sum = 0.0
    

    for j_y, j_x in L: #on parcourt les pixel de la liste L des voisinages les plus proches
        
      if j_x<=width and j_y<=height :
                    # Définir les limites du patch centrée sur (j_x,j_y)
                    start_j_x, end_j_x =  j_x - half_window,  j_x + half_window + 1
                    start_j_y, end_j_y =  j_y - half_window,  j_y + half_window + 1
                    N_j = image[start_j_y:end_j_y, start_j_x:end_j_x]
                    
                    
    
                    # Calculer la distance euclidienne
                    distance_squared = np.sum(kernel *(N_i - N_j) *(N_i - N_j))
                    exponent = distance_squared 
                    total_sum = exponent+total_sum
    # formule finale de h(i)
    h=math.sqrt(total_sum/(Nbr_pixels - 1/wmeme))
    return h


# Neighborhood window size
f = 2
# Similarity window size
t = 5

# Making gaussian kernel
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

# Adding noise into the image
noisex = np.random.normal(0, 0.1, img.shape).astype(np.float64)
noisy = img + noisex

# Assign a clear output image
cleared = np.zeros_like(img)

# Degree of filtering
h = 0.1

# Replicate boundaries of noisy image
noisy2 = np.pad(noisy, ((f, f), (f, f)), mode='symmetric')


# Calcul de la sortie de chaque pixel en utilisant l'optimisation de h
for i in range(f, noisy2.shape[0] - f):
    for j in range(f, noisy2.shape[1] - f):
        W1 = noisy2[i - f:i + f + 1, j - f:j + f + 1]  # Voisinage du pixel concerné (fenêtre de similarité)
        NL = 0
        Z = 0
        L = extract_closest_neighbors(j, i, noisy2, f, 11, t)  # Extraction des voisins les plus proches
        h = calculate_h(j, i, L, noisy2, 0.1, f)  # Calcul de h approprié pour le pixel (i,j)
        
        for r, s in L:  # Parcours des pixels dont le voisinage est le plus proche du pixel (i,j)
            W2 = noisy2[r - f:r + f + 1, s - f:s + f + 1]  # Voisinage du pixel (s,r)
            d2 = np.sum(kernel * (W1 - W2) * (W1 - W2))  # Calcul de la distance euclidienne
            sij = np.exp(-d2 / (h * h))
            Z = Z + sij
            NL = NL + (sij * noisy2[r, s])
        
        cleared[i - f, j - f] = NL / Z  # Sortie du pixel
