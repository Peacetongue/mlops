import tensorflow as tf
import bentoml
import tf2onnx

model = tf.keras.models.load_model("cifar10_model.keras")

spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)

model.output_names=['output']
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

bentoml.onnx.save_model(
    "cifar10_onnx_model",
    model=onnx_model,
    signatures={
        "run": {"batchable": True}
    }
)