from deep_emotion_recognition import DeepEmotionRecognizer
import json
import os
import time
import subprocess

sharedPath = r'C:\Users\R1nge\Documents\TELEGRAM\SHARED'
statePath = os.path.join(sharedPath, "STATE")
statePathJson = os.path.join(statePath, "STATE.json")
faceEmotions = os.path.join(sharedPath, "Emotions_Face.bat")

# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
deeprec.train()
print(deeprec.test_score())


results = []

while True:
    with open(statePathJson, 'r', encoding='utf-8') as f:
        contents = f.read()
        state = json.loads(contents)
        print(state)

    if state['state'] == "splitting_finished":
        state = {
            "state": "detection_emotions"
        }

        media_files = os.listdir(sharedPath)

        with open(statePathJson,'w', encoding='utf-8') as f:
            json.dump(state, f,ensure_ascii=False, indent=4)

        for file in media_files:
            if file.endswith(".oga"):
                # Analyze each file and save to a JSON
                prediction = deeprec.predict(os.path.join(sharedPath, file))
                print(f"Prediction for {file}: {prediction}")
                data = {
                    "emotion": prediction
                }
                results.append(data)

            if file.endswith(".mp4"):
                prediction = deeprec.predict(os.path.join(sharedPath, file))
                print(f"Prediction for {file}: {prediction}")
                data = {
                    "emotion": prediction
                }
                results.append(data)

        with open(os.path.join(sharedPath, "emotions.json"),'w', encoding='utf-8') as f:
            json.dump(results, f,ensure_ascii=False, indent=4)

        time.sleep(2)
        result = subprocess.run([faceEmotions], shell=True)
        print(result)
    else:
        print("waiting")
        time.sleep(1)