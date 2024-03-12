from model_framework import Modeler
from transformers import AutoTokenizer

if __name__ == "__main__":

    print("Analyzing text...")

    # using pretrained model (faster and more lightweight than bert? Can change)
    model_name = "DistilBERT"

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create modeler
    model_wrapper = Modeler(model_name=model_name, tokenizer=tokenizer)

    text = input("input text to analyze: ")

    while text:
        positive, confidence = model_wrapper.evaluate(text)
        if positive:
            print(f"Positive sentiment detected. {confidence} confidence.")
        else:
            print(f"Negative sentiment detected. {confidence} confidence.")
        text = input("input text to analyze: ")
