from deepface import DeepFace

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

alignment_modes = [True, False]

#split video into images, 1 image per 950ms
#analyze each image
#save to a file
objs = DeepFace.analyze(
  img_path = "img.jpg", 
  actions = ['age', 'gender', 'race', 'emotion']
)

print(objs)