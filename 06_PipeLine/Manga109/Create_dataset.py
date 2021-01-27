# This is some code to create the YOLO training data that I will need for my YOLO model.
import manga109api
from PIL import Image, ImageDraw

manga109_root_dir = "Data"
p = manga109api.Parser(root_dir=manga109_root_dir)

def draw_rectangle(img, x0, y0, x1, y1, annotation_type):
    if annotation_type in ["face"]:
    # assert annotation_type in ["body", "face", "frame", "text"]
        color = {"body": "#258039", "face": "#f5be41",
             "frame": "#31a9b8", "text": "#cf3721"}[annotation_type]
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)

def get_faces_for_yolo():
    manga109_root_dir = "Data"
    p = manga109api.Parser(root_dir=manga109_root_dir)
    j = 0
    for book_title in p.books:
        annotation = p.get_annotation(book=book_title)

        for i in range(0, len(annotation['page'])):
            
            img = Image.open(p.img_path(book=book_title, index=i))

            for annotation_type in ["face"]:
                
                j += 1
                rois = annotation["page"][i][annotation_type]

                f = open(f"Yolo_dataset/{j}.txt","w+")  

                for roi in rois:
                    w = (roi["@xmax"] - roi["@xmin"])
                    h = (roi["@ymax"] - roi["@ymin"])
                    x = roi['@xmin'] + ((roi["@xmax"] - roi["@xmin"]) / 2)
                    y = roi["@ymin"] + ((roi["@ymax"] - roi["@ymin"]) / 2)

                    x = round(x / img.size[0], 6)
                    y = round(y / img.size[1], 6)
                    w = round(w / img.size[0], 6)
                    h = round(h / img.size[1], 6)

                    f.write(f"1 {x} {y} {w} {h}\n") 
                f.close()  
                img.save(f"Yolo_dataset/{j}.jpg")


get_faces_for_yolo()