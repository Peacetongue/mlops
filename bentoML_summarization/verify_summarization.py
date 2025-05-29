import requests
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

text = "Высота башни составляет 324 метра (1063 фута), примерно такая же высота, как у 81-этажного здания, и самое высокое сооружение в Париже. Его основание квадратно, размером 125 метров (410 футов) с любой стороны. Во время строительства Эйфелева башня превзошла монумент Вашингтона, став самым высоким искусственным сооружением в мире, и этот титул она удерживала в течение 41 года до завершения строительство здания Крайслер в Нью-Йорке в 1930 году. Это первое сооружение которое достигло высоты 300 метров. Из-за добавления вещательной антенны на вершине башни в 1957 году она сейчас выше здания Крайслер на 5,2 метра (17 футов). За исключением передатчиков, Эйфелева башня является второй самой высокой отдельно стоящей структурой во Франции после виадука Мийо."

model_name = "cointegrated/rut5-base-absum"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
start_local = time.time()
outputs = model.generate(**inputs, max_length=100)
local_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
local_time = time.time() - start_local

print(f"Локальный результат: {local_result}")
print(f"⏱Время локального инференса: {local_time:.3f} сек")

start_remote = time.time()
response = requests.post(
    "http://localhost:3000/summarize",
    json={"text": text, "max_length": 100}
)
remote_time = time.time() - start_remote

if response.status_code == 200:
    remote_result = response.text
    print(f"Ответ от сервиса: {remote_result}")
    print(f"Время удалённого инференса: {remote_time:.3f} сек")
else:
    print(f"Ошибка сервиса: {response.status_code} — {response.text}")


print("\nСравнение результатов:")
print("Совпадают ли тексты?", local_result.strip() == remote_result.strip())
print("Разница во времени (сек):", abs(local_time - remote_time))
