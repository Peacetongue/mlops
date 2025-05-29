import bentoml
from transformers import T5ForConditionalGeneration, T5Tokenizer
from prometheus_client import Counter, Histogram
import time

REQUEST_COUNT = Counter(
    "summary_requests_total",
    "Total number of summarization requests",
)
REQUEST_LATENCY = Histogram(
    "summary_request_latency_seconds",
    "Latency of summarization requests",
)


@bentoml.service(
    resources={"gpu": True},
    traffic={"timeout": 60},
)
class SummarizationService:
    def __init__(self):
        self.model_name = "cointegrated/rut5-base-absum"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    @bentoml.api
    def summarize(self, text: str, max_length: int = 100) -> str:
        start_time = time.time()
        REQUEST_COUNT.inc()

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

        REQUEST_LATENCY.observe(time.time() - start_time)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)