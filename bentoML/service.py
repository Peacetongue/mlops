import bentoml
from bentoml.io import Image, JSON
from bentoml.models import BentoModel
import numpy as np
from PIL import Image as PILImage
import onnxruntime as ort

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@bentoml.service(
    resources={"gpu": True},
    traffic={"timeout": 60}
)
class Cifar10Service:
    model_ref = BentoModel("cifar10_onnx_model:latest")

    def __init__(self):
        self.session = ort.InferenceSession(self.model_ref.path_of("saved_model.onnx"))
        self.input_name = self.session.get_inputs()[0].name

    @bentoml.api
    def classify(self, image: PILImage.Image):
        img = image.resize((32, 32)).convert("RGB")
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        output = self.session.run(None, {self.input_name: arr})[0]
        class_id = int(np.argmax(output))
        confidence = float(np.max(output))

        return {
            "class_id": class_id,
            "class_name": class_names[class_id],
            "confidence": confidence
        }
