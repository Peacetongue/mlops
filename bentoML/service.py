import bentoml
from bentoml.io import Image, JSON
import numpy as np
from PIL import Image as PILImage

model_ref = bentoml.keras.get("cifar10_cnn:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service("cifar10_classifier", runners=[model_runner])

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@svc.api(input=Image(), output=JSON())
async def classify(input_image: PILImage.Image):
    image = input_image.resize((32, 32)).convert("RGB")
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = await model_runner.async_run(image_array)
    predicted_class = int(np.argmax(prediction))
    class_name = class_names[predicted_class]

    return {
        "predicted_class_id": predicted_class,
        "class_name": class_name,
        "confidence": float(np.max(prediction))
    }
