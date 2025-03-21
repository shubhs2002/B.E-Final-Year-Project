import os
import random
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

# Load images and convert them to feature vectors
def load_images(image_folder):
    images = []
    filenames = sorted(os.listdir(image_folder))  # Ensure sorted order
    for file in filenames:
        img_path = os.path.join(image_folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (100, 100)).flatten()  # Resize and flatten
        images.append(img)
    return np.array(images), filenames

# Reduce dimensionality with PCA
def reduce_dimensionality(image_vectors, n_components=50):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(image_vectors), pca

def find_closest_stage(input_image,user_img_vector, stored_vectors, stage_distribution):
    similarities = cosine_similarity(user_img_vector.reshape(1, -1), stored_vectors)[0]
    closest_img_index = np.argmax(similarities)  # Get the most similar image index

    # Convert closest index to filename (since images are stored as Img1.png, Img2.png, ...)
    closest_img_name = f"Img{closest_img_index + 1}.png"  # Adding +1 to align with filename numbering
    print(closest_img_name)
    # Find which stage this closest image belongs to
    for stage in range(len(stage_distribution)):
        for image in stage_distribution[stage]:
            if image in input_image:
                return stage
    return -1  # Return None if no match found

main_folder = "D:\\A DATA\\Project Module 2021-22\\A PYTHON New\\Plant Growth Prediction\\CompleteUpdatedPlantGrowth\\" 
#D:\\A DATA\\Project Module 2021-22\\A PYTHON New\\Plant Growth Prediction\\CompleteUpdatedPlantGrowth 
# Main function
def main(input_image):
    try:
        closest_stage = 0
        print(input_image)
        stages_resutl = {}
        images, filenames = load_images(main_folder+"static\\train")
        total_images = len(filenames)  # Dynamically set total images
        total_stages = 3

        # Compute base number of images per stage (some will have 4, some 5)
        images_per_stage = total_images // total_stages  # 4
        extra_images = total_images % total_stages  # 8 (first 8 stages get 5 images)
        image_names = [f"Img{i}.png" for i in range(1, total_images + 1)]
        stage_distribution = []
        start = 0
        for i in range(total_stages):
            num_images = images_per_stage + (1 if i < extra_images else 0)  # First 8 stages get 5 images, rest get 4
            stage_distribution.append(image_names[start:start + num_images])
            start += num_images

        # Reduce image vectors using PCA
        reduced_images, pca_model = reduce_dimensionality(images)
        
        # Load user-provided image
        user_img_path = main_folder+input_image
        user_img = cv2.imread(user_img_path, cv2.IMREAD_GRAYSCALE)
        user_img = cv2.resize(user_img, (100, 100)).flatten()
        user_img_vector = pca_model.transform(user_img.reshape(1, -1))  # Apply PCA transformation

        closest_stage = find_closest_stage(input_image,user_img_vector, reduced_images, stage_distribution)
        if closest_stage+1==1:
            stages_resutl = {"Stage 1 Result":random.choice(stage_distribution[0]),"Stage 2 will be":random.choice(stage_distribution[1]),"Stage 3 Result":random.choice(stage_distribution[2])}
        elif closest_stage+1==2:
            stages_resutl = {"Stage 2 Result":random.choice(stage_distribution[1]),"Stage 3 will be":random.choice(stage_distribution[2])}
        elif closest_stage+1==3:
            stages_resutl = {"Stage 3 Result":random.choice(stage_distribution[2])}
        
    except Exception as e:
        print(e)
    
    return closest_stage+1,stages_resutl

# print(main('static/input\Img20.png'))