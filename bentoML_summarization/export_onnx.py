from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = "cointegrated/rut5-base-absum"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

dummy_input = tokenizer(
    "Пример текста для суммаризации",
    return_tensors="pt",
    max_length=512,
    truncation=True
)

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "rut5-summarization.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
    },
    opset_version=14,
)