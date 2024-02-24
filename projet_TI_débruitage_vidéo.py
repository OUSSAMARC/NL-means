import imageio
import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
import os




script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir)) 
video_path = os.path.join(parent_directory, 'Images et vidéos', 'output_bruitage_video.mp4') 


# Lecture de la vidéo avec imageio
video_reader = imageio.get_reader(video_path)

# Obtention des propriétés de la vidéo
frame_width = video_reader.get_meta_data()['size'][0]
frame_height = video_reader.get_meta_data()['size'][1]

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
        ab = x - f
        cd = y - f
        ker[x, y] = 10 * np.exp(((ab * ab) + (cd * cd)) / (-2 * (su * su)))
        sm = sm + ker[x, y]

kernel = ker / f
kernel = kernel / sm

# Fonction pour appliquer l'algorithme à chaque canal
def apply_algorithm_to_channel(channel):   
    # Assignation d'une image de sortie claire
    cleared = np.zeros_like(channel)
    # Degré de filtrage
    h = 0.07
    # Réplication des bordures de l'image bruitée
    noisy2 = np.pad(channel, ((f, f), (f, f)), mode='symmetric')

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
    
    return cleared

# On remplira cette liste des frames après NL means
L = []

# Boucle de traitement des frames de la vidéo
for i, frame in enumerate(video_reader):
    
    frame = np.array(frame) / 255.0
    
    filtered_channels = []
    img_channels = [frame[:, :, j] for j in range(3)]

    # Apply the algorithm to each channel and append to filtered_channels
    for channel in img_channels:
        filtered_channels.append(apply_algorithm_to_channel(channel))
    filtered_img_color = np.stack(filtered_channels, axis=-1)
    
    L.append(filtered_img_color)


"""
La partie restante du code est dédiée à la sauvegarde de la vidéo
"""
    
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir)) 
image_folder = os.path.join(parent_directory, 'Images et vidéos', 'temp_images') 

os.makedirs(image_folder, exist_ok=True)

for i, frame in enumerate(L):
    image_path = os.path.join(image_folder, f"frame_{i:03d}.png")
    plt.imsave(image_path, np.clip(frame, 0, 1))

# Créer la vidéo à partir des images sauvegardées
fps = 20
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)

output_video_folder = os.path.join(parent_directory, 'Images et vidéos Résultats')
output_video_path = os.path.join(output_video_folder, 'output_debruitagee_video.mp4')

clip.write_videofile(output_video_path)

# Nettoyer le répertoire temporaire après la création de la vidéo
for file in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file)
    os.remove(file_path)
os.rmdir(image_folder)
