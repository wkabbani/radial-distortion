from typing import Any, Callable, Tuple

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

from processing.utils import pil_to_mediapipe


def get_facial_detector(detector_path: str):
    base_options = python.BaseOptions(model_asset_path=detector_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    return detector


def get_facial_landmarks_detector(detector_path: str):
    base_options = python.BaseOptions(model_asset_path=detector_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector_landmark = vision.FaceLandmarker.create_from_options(options)
    return detector_landmark


def mediapipe_face_detector(
    detector_path: str,
) -> Callable[..., Tuple[int, int, int, int]]:
    def detection(image) -> Tuple[int, int, int, int]:
        detector = get_facial_detector(detector_path=detector_path)
        detection_result = detector.detect(image)

        if len(detection_result.detections) == 0:
            return 0, 0, image.width, image.height

        bbox = detection_result.detections[0].bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        return (start_point[0], start_point[1], end_point[0], end_point[1])

    return detection


def mediapipe_face_landmarks_detector(detector_path):
    landmarks_detector = get_facial_landmarks_detector(detector_path=detector_path)

    def detection(image: Image.Image):
        mediapipe_img = pil_to_mediapipe(image)
        detection_result = landmarks_detector.detect(mediapipe_img)

        if len(detection_result.face_landmarks) == 0:
            return None

        return detection_result

    return detection
