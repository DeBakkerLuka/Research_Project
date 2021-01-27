import cv2
import numpy as np
import config 
import os
from PIL import Image, ImageDraw
from skimage.io import imread, imsave, imshow

# Yolo model inladen en klaarzetten voor gebruik
net = cv2.dnn.readNet("Weights_detec/manga_v7_final_weights.weights", "Weights_detec/manga_v7_final_config.cfg")  # weight en configuration file ophalen
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Namen van de classes definiëren (volgorde is zeer belangrijk)
classes = ["RandomFrame", "Face", "Text"]


def draw_and_save(files, directory, j):

    

    for file in files:
        img = cv2.imread(directory + "/" +  file)

        img = np.array(img)

        stacked_img = np.stack((img,)*3, axis=-1)

        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
    
        class_ids, confidences, boxes, centers = [], [], [], []

        image = Image.fromarray(img)
        
        for out in outs:
            
            for detection in out:
               
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
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
            j += 1 
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence = round(confidences[i] * 100, 1)
                label = str(classes[int(class_ids[i])])
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(211, 44, 44), thickness=3)
                cv2.putText(img, str(confidence), (x + 5, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color=(211, 44, 44), thickness=4) 

                w_norm = ((x + w) - x)
                h_norm = ((y + h) - y)
                x_norm = x + (w/2)
                y_norm = y + (h/2)

                image2 = image.crop((x, y, x+w, y+h))

                image2.save(f"Original_dataset/{j}.jpg") 

                


directory = config.volumes_dir
files = os.listdir(directory)
j = 0
chosen_directory = 0
while chosen_directory != 'exit':
    print("Which volume of manga would you like to use detection on?")
    for i in range(0, len(files)):
        print(f" \t {i}. {files[i]}")

    chosen_volume = input("Which volume of manga would you like to use detection on?")
    chosen_directory = directory + "\\" + str(files[int(chosen_volume)])
    draw_and_save(os.listdir(chosen_directory), chosen_directory, j)
    j += 500