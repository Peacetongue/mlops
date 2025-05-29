import requests

def test_summarization():
    response = requests.post(
        "http://localhost:3000/summarize",
        json={"text": "Москва — столица России, крупнейший город страны...", "max_length": 50}
    )
    assert response.status_code == 200
    assert len(response.json()) > 0
    print("Test passed! Summary:", response.json())

if __name__ == "__main__":
    test_summarization()