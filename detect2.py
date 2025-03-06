from deepface import DeepFace
import json
import os

sharedPath = r'C:\Users\R1nge\Documents\TELEGRAM\SHARED'
statePath = os.path.join(sharedPath, "STATE")
statePathJson = os.path.join(statePath, "STATE.json")

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

i = 0
for file in os.listdir(sharedPath):
    if file.endswith(".png"):
        # Analyze each file and save to a JSON
        objs = DeepFace.analyze(
            img_path = f"{sharedPath}\{file}", 
            #actions = ['age', 'gender', 'race', 'emotion']
            actions = ['emotion', 'gender']
        )
        print(f"Prediction for {file}: {objs}")
        data = {
            "emotion": objs[0]['dominant_emotion'],
            "gender": objs[0]['dominant_gender']
        }
        print(data)
        with open(os.path.join(sharedPath, f"{i}_face.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        i += 1


state = {
            "state": "detection_finished"
        }
with open(statePathJson,'w', encoding='utf-8') as f:
    json.dump(state, f,ensure_ascii=False, indent=4)

print(objs)