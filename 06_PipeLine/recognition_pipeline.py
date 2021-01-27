# This will be my pipeline that I will show off as a demo.
# This pipeline will take a manga page, crop all the detected faces out of it
# The cropped faces will be put into a recognition model
# The output will be the original manga page, with bounding boxes and names drawn onto it.

# Imports needed for pipeline
import cv2
import os
import config
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray 
from PIL import Image, ImageDraw
from skimage import transform
from skimage.io import imread, imsave, imshow
from tensorflow import keras
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img

# Getting current directory 
dir_path = os.path.dirname(os.path.realpath(__file__))

# Getting all the classes in dataset
directory = 'Original_dataset'
classes_recog = os.listdir(directory)
classes_recog.remove('Readme.md')
print(classes_recog)

model_recog = keras.models.load_model(dir_path + r"/Weights_recog/" + config.save_model_name)

# Yolo model inladen en klaarzetten voor gebruik
net = cv2.dnn.readNet(dir_path + "\Weights_detec\manga_v7_final_weights.weights", dir_path + "\Weights_detec\manga_v7_final_config.cfg")  

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Namen van de classes definiëren (volgorde is zeer belangrijk)
classes = ["RandomFrame", "Face", "Text"]

def drawboxes(files, directory, volume_name):

    if not os.path.exists(f"Results/{str(volume_name)}"):
            os.makedirs(f"Results/{str(volume_name)}")

    j = 0

    for file in files: 

        # Reading and preprocessing the page of the volume.
        img = cv2.imread(directory + "/" +  file)
        img = np.array(img)
        stacked_img = np.stack((img,)*3, axis=-1)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Creating variables, incrementing J for saving images.
        class_ids, confidences, boxes = [], [], []
        
        j += 1

        image = Image.fromarray(img)
        # For every detection in each page.
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # boxes tekenen
        for i in range(len(boxes)):
           
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence = round(confidences[i] * 100, 1)
                label = str(classes[int(class_ids[i])])

                # Drawing the rectangle with an accuracy inside of it.
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(211, 44, 44), thickness=3)
                cv2.putText(img, str(confidence), (x + 5, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.4, color=(211, 44, 44), thickness=4)

                # This image will be used to predict which character is present.
                # So this image will be put into the recognition model.
                image2 = image.crop((x, y, x+w, y+h))

                image2.save("0.jpg")

                image2 = load_img("0.jpg", target_size=(224,224))

                numpyImage = img_to_array(image2)

                numpyImage /= 255

                numpyImage = numpyImage.reshape((1,224,224,3))

                pred = model_recog.predict(numpyImage)

                print(pred[0])

                pred_location = np.argmax(pred[0])

                pred_value = pred[0][pred_location]

                print(pred_value)

                if pred_value >= 0.8:
                    pred_chara = classes_recog[np.argmax(pred[0])]
                elif pred_value < 0.8:
                    pred_chara = "Unknown"

                cv2.putText(img, str(pred_chara), (x + 5, y + 60), cv2.FONT_HERSHEY_PLAIN, 1.3, color=(211, 44, 44), thickness=2)
                
        # Saving the image to a results folder.
        image = Image.fromarray(img)
        image.save(f"Results/{volume_name}/{j}.jpg")


directory = config.volumes_dir
files = os.listdir(directory)
chosen_volume = 0


while chosen_volume != "exit":
    print("Which volume of manga would you like to use detection on?")
    for i in range(0, len(files)):
        print(f" \t {i}. {files[i]}")
    chosen_volume = input("Which volume of manga would you like to use detection on? ")
    chosen_directory = directory + "\\" + str(files[int(chosen_volume)])
    drawboxes(os.listdir(chosen_directory), chosen_directory,  chosen_directory.partition("Volumes\\")[2])