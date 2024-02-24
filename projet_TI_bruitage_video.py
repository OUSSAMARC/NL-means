import imageio
import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
import os


script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir)) 
video_path = os.path.join(parent_directory, 'Images et vidéos', 'Calma.mp4') 

# Lecture de la vidéo avec imageio
video_reader = imageio.get_reader(video_path)

# Obtention des propriétés de la vidéo
frame_width = video_reader.get_meta_data()['size'][0]
frame_height = video_reader.get_meta_data()['size'][1]

L = []

# Boucle de traitement des frames de la vidéo
for i, frame in enumerate(video_reader):
    # Lire la frame actuelle    
    frame = np.array(frame) / 255
    

    noisex = np.random.normal(0, 0.1, frame.shape).astype(np.float64)
    frame = frame + noisex

    L.append(frame)

"""
La partie restante du code est dédiée à la sauvegarde de la vidéo
"""

# Sauvegarder les images dans un répertoire temporaire
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

output_video_folder = os.path.join(parent_directory, 'Images et vidéos')
output_video_path = os.path.join(output_video_folder, 'output_bruitage_video.mp4')

clip.write_videofile(output_video_path)

# Nettoyer le répertoire temporaire après la création de la vidéo
for file in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file)
    os.remove(file_path)
os.rmdir(image_folder)