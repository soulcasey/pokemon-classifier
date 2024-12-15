import torch
import cv2
from torchvision import transforms
import common

transform = common.get_transform()

class_names = torch.load("class_names.pth", weights_only=True)

# Load the model and class names
model = common.get_model(class_names)
model.load_state_dict(torch.load("pokemon_classifier.pth", weights_only=True, map_location=common.get_device()))
model.eval()
device = common.get_device()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = transform(transforms.ToPILImage()(img)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
    
    # Display the frame and prediction
    cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Pokemon Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
