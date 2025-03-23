from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Classifier:
    def __init__(self, model_name="AmirMohseni/BERT-Router-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model_name = model_name

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return self.model.config.id2label[predicted_class]
    
    def get_model_name(self):
        return self.model_name
    
if __name__ == "__main__":
    classifier = Classifier()
    # Run tests
    test_query = "Hello, how are you?"
    result = classifier.classify(test_query)
    expected = "small_llm"
    print(f"Test 1: {test_query} -> {result}")
    print("Test 1 passed!" if result == expected else "Test 1 failed!")
    
    test_query = """
        Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
    \[\log_2\left({x \over yz}\right) = {1 \over 2}\]
    \[\log_2\left({y \over xz}\right) = {1 \over 3}\]
    \[\log_2\left({z \over xy}\right) = {1 \over 4}\]
    Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
    """
    result = classifier.classify(test_query)
    expected = "large_llm"
    print(f"Test 2: {test_query} -> {result}")
    print("Test 2 passed!" if result == expected else "Test 2 failed!")
