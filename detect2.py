from deepface import DeepFace
import json
import os
import subprocess

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
Gaze = os.path.join(sharedPath, "Gaze.bat")
png_files = os.listdir(sharedPath)
results = []

i = 0
for file in png_files:
    if file.endswith(".png"):
        # Analyze each file and save to a JSON
        objs = DeepFace.analyze(
            img_path = f"{sharedPath}\{file}", 
            #actions = ['age', 'gender', 'race', 'emotion']
            actions = ['emotion', 'gender'],
            enforce_detection=False
        )
        print(f"Prediction for {file}: {objs}")
        data = {
            "emotion": objs[0]['dominant_emotion'],
            "gender": objs[0]['dominant_gender']
        }

        results.append(data)
        print(data)
        i += 1


with open(os.path.join(sharedPath, f"face.json"), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

state = {
            "state": "detection_finished"
        }

with open(statePathJson,'w', encoding='utf-8') as f:
    json.dump(state, f,ensure_ascii=False, indent=4)

result = subprocess.run([Gaze], shell=True)

print(objs)