import torch
from wrapper import Wrapper
from transformers import AutoTokenizer, AutoModel
import datasets
from model import Model

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    wrapper = Wrapper(model_name="distilbert-base-uncased")

    
    # Load the IMDb dataset
    imdb_dataset = datasets.load_dataset("imdb")

    # Access the test split dataset instance from the list
    train_set = imdb_dataset["train"]
    train_set = train_set.map(
        wrapper.tokenize, fn_kwargs={"tokenizer": tokenizer}
    )
    train_set = train_set.with_format(type="torch", columns=['ids', 'label'])

    # create data loaders
    test_dataloader = wrapper.data_loader(data=train_set, batch_size=16, pad_index=tokenizer.pad_token_id)

    # Load the saved model
    model = Model(model=AutoModel.from_pretrained("distilbert-base-uncased"), output=len(train_set["label"].unique()), freeze=False)
    model.load_state_dict(torch.load("model.pt"))
    critereon = torch.nn.CrossEntropyLoss()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    text = input("Input text to analyze: ")
    while text:
        print(wrapper.evaluate_text(text, model, tokenizer, device))
        text = input("Input text to analyze: ")