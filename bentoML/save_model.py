from transformers import pipeline
import bentoml

summarizer = pipeline("summarization", model="cointegrated/rut5-base-absum")

bentoml.transformers.save_model("text_summarizer", summarizer)