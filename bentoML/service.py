import bentoml
from bentoml.io import JSON

model_ref = bentoml.transformers.get("text_summarizer:latest")
summarizer_runner = model_ref.to_runner()

svc = bentoml.Service("summarizer_service", runners=[summarizer_runner])

@svc.api(input=JSON(), output=JSON())
async def summarize(input_json: dict):
    text = input_json["text"]
    result = await summarizer_runner.async_run(text)
    return {"summary": result[0]["summary_text"]}
