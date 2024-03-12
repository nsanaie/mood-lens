import pandas as pd
import torch
from torch.utils.data import Dataset

class TextAndLabelsSet(Dataset):

    def __init__(self, path, tokenizer, max_length=128):

        # init the dataset
        self.df = pd.read_csv(path, delimiter="\t")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.df['sentence'].tolist()
        self.labels = self.df['label'].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # tokenize the text
        # encoding = self.tokenizer.encode_plus(
        #     text,
        #     add_special_tokens=True,
        #     max_length=self.max_length,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt'
        # )

        # input_ids = encoding['input_ids'].squeeze()
        # attention_mask = encoding['attention_mask'].squeeze()


        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        if len(tokens) < self.max_length:
            tokens = tokens + ["[PAD]" for _ in range(self.max_length - len(tokens))]
        else:
            tokens = tokens[: self.max_length - 1] + ["[SEP]"]
        
        # Obtain indices of tokens and convert them to tensor.
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        attention_mask = (input_ids != 0).long()

        return input_ids, attention_mask, label