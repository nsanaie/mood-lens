import datasets 
from wrapper import Wrapper
from transformers import AutoTokenizer

if __name__ == "__main__":

    # model name (can be changed depending on use case). Using BERT for testing
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    wrapper = Wrapper(model_name)

    # load training/validation sets using huggingface imdb dataset
    train_set, test_set = datasets.load_dataset("imdb", split=["train", "test"])
    train_set = train_set.map(
        Wrapper.tokenize, fn_kwargs={"tokenizer": tokenizer}
    )
    test_set = test_set.map(
        Wrapper.tokenize, fn_kwargs={"tokenizer": tokenizer}
    )

    valid_data = train_set.train_test_split(test_size=0.25)
    train_set = valid_data['train'].with_format(type="torch", columns=['ids', 'label'])
    validation_set = valid_data['test'].with_format(type="torch", columns=['ids', 'label'])
    test_set = test_set.with_format(type="torch", columns=['ids', 'label'])

    # create data loaders
    train_dataloader = Wrapper.data_loader(data=train_set, batch_size=16, pad_index=tokenizer.pad_token_id)
    test_dataloader = Wrapper.data_loader(data=test_set, batch_size=16, pad_index=tokenizer.pad_token_id)
    validation_dataloader = Wrapper.data_loader(data=validation_set, batch_size=16, pad_index=tokenizer.pad_token_id)