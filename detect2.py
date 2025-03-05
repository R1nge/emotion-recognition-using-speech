from deepface import DeepFace
import json
import os

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yolov11s',
  'yolov11n',
  'yolov11m',
  'yunet',
  'centerface',
]

sharedPath = r'C:\Users\R1nge\Documents\TELEGRAM\SHARED'

#split video into images, 1 image per 950ms
#analyze each image
#save to a file

i = 0
for file in os.listdir(sharedPath):
    if file.endswith(".mp4"):
        # Analyze each file and save to a JSON
        objs = DeepFace.analyze(
            img_path = "img.jpg", 
            actions = ['age', 'gender', 'race', 'emotion']
        )
        print(f"Prediction for {file}: {objs}")
        data = objs[0]['dominant_emotion']
        print(data)
        with open(os.path.join(sharedPath, f"{i}_face.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        i += 1




print(objs)