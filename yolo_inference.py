from ultralytics import YOLO

model = YOLO('models\cvFootball-Best.pt')

results = model.predict('imported videos/test (17).mp4',save= True)
print(results[0])

print('================================')
for box in results[0].boxes:
    print(box)