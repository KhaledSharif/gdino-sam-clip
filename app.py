from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill.core.composed_detection_model import ComposedDetectionModel
import cv2

classes = ["McDonalds", "Burger King"]


SAMCLIP = ComposedDetectionModel(
    detection_model=GroundedSAM(
        CaptionOntology({"logo": "logo"})
    ),
    classification_model=CLIP(
        CaptionOntology({k: k for k in classes})
    )
)

IMAGE = "logo.jpg"

results = SAMCLIP.predict(IMAGE)

results = list(zip(results.confidence, results.class_id, results.xyxy))

for r in results:
    print(r)

# Output:
# (0.47160587, 0, array([ 806.93274,  182.77536, 1373.5729 ,  392.29868], dtype=float32))
# (0.3557948, 0, array([1279.0231 ,  193.7986 , 1360.0944 ,  266.98102], dtype=float32))

image = cv2.imread(IMAGE)
# Define font properties for text labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (255, 255, 255)  # White color in BGR

# Draw text labels for each bounding box
for confidence, class_id, bbox in results:
    x1, y1, x2, y2 = map(int, bbox)  # Convert coordinates to integers
    
    # Format the text label
    text_label = f"{classes[class_id]}: {confidence:.2f}"
    
    # Calculate the size of the text label
    (text_width, text_height), _ = cv2.getTextSize(text_label, font, font_scale, font_thickness)
    
    # Calculate the position of the text label
    text_x = x1
    text_y = max(y1 - 10, 0) 
    
    cv2.putText(image, text_label, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    cv2.rectangle(image, (x1, text_y - text_height), (x1 + text_width, text_y), (0,255,0), 1)

cv2.imwrite("logo_annotated.jpg", image)