import time
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
import io

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = y_test.flatten()

N = 100
x_sample = x_test[:N]
y_sample = y_test[:N]

local_model = tf.keras.models.load_model("cifar10_model.keras")

start_local = time.time()
local_preds = local_model.predict(x_sample)
local_classes = np.argmax(local_preds, axis=1)
local_time = time.time() - start_local
local_acc = accuracy_score(y_sample, local_classes)

remote_classes = []
start_remote = time.time()

for img in x_sample:
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    files = {'image': ('image.png', img_buffer, 'image/png')}
    response = requests.post("http://localhost:3000/classify", files=files)

    try:
        result = response.json()
        remote_classes.append(result["class_id"])
    except Exception as e:
        print("Ошибка декодирования JSON:", e)
        print("Ответ сервера:", response.text)
        remote_classes.append(-1)


remote_time = time.time() - start_remote
remote_acc = accuracy_score(y_sample, remote_classes)

print("Точность локальной модели:", f"{local_acc:.3f}")
print("Точность модели после сервинга:", f"{remote_acc:.3f}")
print("Время инференса локально:", f"{local_time:.2f} сек")
print("Время инференса через API:", f"{remote_time:.2f} сек")
print("Совпадают ли предсказания:", np.array_equal(local_classes, remote_classes))
