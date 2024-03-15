import datasets 
from wrapper import Wrapper
from transformers import AutoTokenizer, AutoModel
from model import Model
import torch
import tqdm

if __name__ == "__main__":

    # model name (can be changed depending on use case). Using BERT for testing
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    wrapper = Wrapper(model_name=model_name)

    # load training/validation sets using huggingface imdb dataset
    train_set, test_set = datasets.load_dataset("imdb", split=["train", "test"])
    train_set = train_set.map(
        wrapper.tokenize, fn_kwargs={"tokenizer": tokenizer}
    )
    test_set = test_set.map(
        wrapper.tokenize, fn_kwargs={"tokenizer": tokenizer}
    )

    valid_data = train_set.train_test_split(test_size=0.25)
    train_set = valid_data['train'].with_format(type="torch", columns=['ids', 'label'])
    validation_set = valid_data['test'].with_format(type="torch", columns=['ids', 'label'])
    test_set = test_set.with_format(type="torch", columns=['ids', 'label'])

    # create data loaders
    train_dataloader = wrapper.data_loader(data=train_set, batch_size=16, pad_index=tokenizer.pad_token_id)
    test_dataloader = wrapper.data_loader(data=test_set, batch_size=16, pad_index=tokenizer.pad_token_id)
    validation_dataloader = wrapper.data_loader(data=validation_set, batch_size=16, pad_index=tokenizer.pad_token_id)

    # create model to fine tune
    model = Model(model=AutoModel.from_pretrained(model_name), output=len(train_set["label"].unique()), freeze=False)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # optimzer and critereon for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    critereon = torch.nn.CrossEntropyLoss()

    # create and load model to the device (gpu if possible)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    critereon = critereon.to(device)

    # training model
    epochs = 3
    best_loss = -1000000

    for i in tqdm.trange(epochs):
        train_acc, train_loss = wrapper.train_model(train_dataloader, model, critereon, optimizer, device)
        valid_acc, valid_loss = wrapper.evaluate_model(validation_dataloader, model, critereon, optimizer, device)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "model.pt")
        print(f"epoch: {i}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")