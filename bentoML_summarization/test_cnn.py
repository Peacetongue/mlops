import requests
from PIL import Image
import io

image = Image.open("example.png").resize((32, 32)).convert("RGB")

buf = io.BytesIO()
image.save(buf, format="PNG")
buf.seek(0)

resp = requests.post("http://localhost:3000/classify", files={"image": ("image.png", buf, "image/png")})
print(resp.status_code)
print(resp.json())
