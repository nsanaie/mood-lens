import torch
from torch.utils.data import DataLoader
from model import Model
from transformers import AutoModel

class Wrapper:

    def __init__(self, model_name):

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
    
    def tokenize(text, tokenizer):
        """
        Tokenize an example text.
        """
        return {"ids": tokenizer(text['text'], truncation=True)["input_ids"]}
    
    def data_loader(data, batch_size, pad_index):
        """
        Create DataLoader to retreive batches of examples
        """

        def collate_function(batch):
            """
            Collate function for DataLoader
            """
            ids = [b['id'] for b in batch]
            ids = torch.nn.utils.rnn.pad_packed_sequence(batch_ids, padding_value=pad_index, batch_first=True)
            label = [b['id'] for b in batch]
            label = torch.stack(label)
            return {
                "ids": ids, 
                "label": label
            }


        return DataLoader(
            data=data,
            batch_size=batch_size,
            collate_fn=collate_function,
            shuffle=False
        )