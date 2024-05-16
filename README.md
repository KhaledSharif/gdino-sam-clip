# gdino-sam-clip

Use Grounding DINO, Segment Anything, and CLIP to label objects in images

---

### Requirements

- `autodistill`, `autodistill_clip`, and `autodistill_grounded_sam` are used for the CLIP and GroundedSAM models
- `supervision` is used for image annotation
- `cv2` is OpenCV, a library for image processing (`pip install opencv-python`)

### Detection Model

```python
SAMCLIP = ComposedDetectionModel(
    detection_model=GroundedSAM(
        CaptionOntology({"logo": "logo"})
    ),
    classification_model=CLIP(
        CaptionOntology({k: k for k in classes})
    )
)
```

- *Detection*: The GroundedSAM model, using `CaptionOntology({"logo": "logo"})`, scans the image to find anything that looks like a logo. It doesn't need to differentiate between different types of logos at this stage—just find potential logo candidates.

- *Classification*: Once the potential logos are detected, the CLIP model, using `CaptionOntology({k: k for k in classes})`, takes over to classify these detected logos into specific categories ("McDonalds" or "Burger King").

### Annotate Image

```python
image = cv2.imread("IMAGE.jpg")
annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{classes[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _ in results
]

annotated_frame = annotator.annotate(
    scene=image.copy(), detections=results
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame, labels=labels, detections=results
)

sv.plot_image(annotated_frame, size=(8, 8))
```
