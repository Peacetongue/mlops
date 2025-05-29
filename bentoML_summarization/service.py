import base64
from io import BytesIO
import bentoml
from transformers import T5ForConditionalGeneration, T5Tokenizer
from prometheus_client import Counter, Histogram
import time
import numpy as np
from PIL import Image
from bentoml.io import JSON
from pydantic import BaseModel


class ImageInput(BaseModel):
    image: str

classify_input = JSON(pydantic_model=ImageInput)
classify_output = JSON()


class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


request_counter = Counter(
    name='summary_requests_total',
    documentation='Total number of summarization requests',
    labelnames=['status']
)

inference_time_histogram = Histogram(
    name='inference_time_seconds',
    documentation='Time taken for summarization inference',
    labelnames=['status'],
    buckets=(0.1, 0.2, 0.5, 1, 2, 5, 10, float('inf'))
)


@bentoml.service(
    resources={"gpu": True},
    traffic={"timeout": 60},
)
class SummarizationService:
    def __init__(self):
        self.model_ref = bentoml.onnx.get("cifar10_onnx_model:latest")
        self.cnn_session = self.model_ref.to_runner().model
        self.cnn_input_name = self.cnn_session.get_inputs()[0].name
        self.model_name = "cointegrated/rut5-base-absum"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    @bentoml.api
    def classify(self, image: str):
        start_time = time.time()
        status = "success"

        try:
            img_bytes = base64.b64decode(image)
            img = Image.open(BytesIO(img_bytes)).resize((32, 32)).convert("RGB")

            arr = np.array(img).astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            output = self.cnn_session.run(None, {self.cnn_input_name: arr})[0]
            class_id = int(np.argmax(output))
            confidence = float(np.max(output))

            return {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": confidence
            }

        except Exception as e:
            status = "failure"
            return {"error": str(e)}

        finally:
            request_counter.labels(status=status).inc()
            inference_time_histogram.labels(status=status).observe(time.time() - start_time)


    @bentoml.api
    def summarize(self, text: str, max_length: int = 100) -> str:
        start_time = time.time()

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            outputs = self.model.generate(
                **inputs,
                max_length=max_length
            )
            status = 'success'
            summary_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            summary_text = str(e)
            status = 'failure'
        finally:
            inference_time_histogram.labels(status=status).observe(time.time() - start_time)
            request_counter.labels(status=status).inc()

        return summary_text
