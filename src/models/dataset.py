import pandas as pd
from torch.utils.data import Dataset

class TextAndLabelsSet(Dataset):

    def __init__(self, path, tokenizer, max_length=128):

        # init the dataset
        self.df = pd.read_csv(path)
        
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
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, label