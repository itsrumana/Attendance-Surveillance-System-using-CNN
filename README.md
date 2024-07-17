
# Attendance-Surveillance-System-using-CNN

This project introduces an Attendance Surveillance System using Convolutional Neural Networks (CNNs) for accurate attendance monitoring. It collects a classroom dataset, detects faces with MTCNN, and splits them for training (367 images) and testing (24 images). FaceNet's transfer learning improves recognition, achieving an 87.5% accuracy, surpassing SVM's 83.3%. Implemented with Tkinter, the system offers user-friendly functionalities like uploading images, executing recognition, and displaying results in an Excel file. CNNs streamline attendance tracking, ensuring efficiency, and reliability in educational settings.


## Vedio Demonstration

https://github.com/user-attachments/assets/66f12209-3379-4a9d-9ce8-d70a768763c6

## Dataset

The dataset utilized in this project was collected manually by myself and my peers, where we voluntarily captured photos of our classroom using a basic phone. Due to privacy and security concerns, I am not able to share this dataset. However, if you intend to implement this project, you can capture photos of your own classroom and store them in a folder named, for example, "screen2".
## Face Extraction

This code was used to extract individual faces of each student from the classroom image.

```bash
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import os
from keras.preprocessing.image import ImageDataGenerator

def save_face(img, box, folder_path, face_id):
    x, y, w, h = box
    face = img[y:y+h, x:x+w]
    face_path = os.path.join(folder_path, f'face_{face_id}.jpg')
    cv.imwrite(face_path, cv.cvtColor(face, cv.COLOR_RGB2BGR))
    return face_path

def augment_and_save(image_path, output_folder, datagen, face_id):
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) 
    image_rgb = image_rgb.reshape((1, ) + image_rgb.shape)
    i = 0
    for batch in datagen.flow(image_rgb, batch_size=1, save_to_dir=output_folder, save_prefix=f'face_{face_id}_aug', save_format='jpg'):
        i += 1
        if i >= 5:  
            break

img_path = 'screen2/six.jpeg'
img = cv.imread(img_path)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

detector = MTCNN()
results = detector.detect_faces(img_rgb)

output_folder = 'test'
os.makedirs(output_folder, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

for i, result in enumerate(results):
    box = result['box']
    face_path = save_face(img_rgb, box, output_folder, i)
    augment_and_save(face_path, output_folder, datagen, i)

print("Faces extracted and augmented images saved.")

```
Next, create two folders named "train" and "test" and organize the extracted images based on the desired ratio for training and testing purposes.

## Accuracy Achieved

SVM-83.3% <br>
CNN-87.5%
