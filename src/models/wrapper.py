import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np

class Wrapper:

    def __init__(self, model_name):

        self.model_name = model_name
    
    def tokenize(self, text, tokenizer):
        """
        Tokenize an example text.
        """
        return {"ids": tokenizer(text['text'], truncation=True)["input_ids"]}
    
    def data_loader(self, data, batch_size, pad_index):
        """
        Create DataLoader to retreive batches of examples
        """

        def get_collate_fn(pad_index):
            def collate_fn(batch):
                batch_ids = [i["ids"] for i in batch]
                batch_ids = torch.nn.utils.rnn.pad_sequence(
                    batch_ids, padding_value=pad_index, batch_first=True
                )
                batch_label = [i["label"] for i in batch]
                batch_label = torch.stack(batch_label)
                batch = {"ids": batch_ids, "label": batch_label}
                return batch

            return collate_fn


        return DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=get_collate_fn(pad_index),
            shuffle=False
        )

    def accuracy_check(self, pred, label):
        """
        Helper function for calculating prediction accuracy
        """
        size = pred.shape[0]
        pred_classes = pred.argmax(dim=-1)
        pred_correct = pred_classes.eq(label).sum()
        return pred_correct / size

    def train_model(self, data, model, crit, opt, dev):
        """
        Algo for training the model. Can add lists here for data logging losses and accuracies if there are issues in training
        """
        # set to training mode
        model.train()

        accs = []
        losses = []
        
        for b in tqdm.tqdm(data, desc="Training Model: "):
            ids = b['ids'].to(dev)
            label = b['label'].to(dev)
            
            pred = model(ids)
            loss = crit(pred, label)
            acc = self.accuracy_check(pred, label)
            accs.append(acc.item())
            losses.append(loss.item())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return np.mean(accs), np.mean(losses)

    def evaluate_model(self, data, model, crit, dev):
        """
        Algo for evaluating the model after training 
        """

        # set to eval mode
        model.eval()

        accs = []
        losses = []

        with torch.no_grad():
            for b in tqdm.tqdm(data, desc="Evaluating Model: "):
                ids = b['ids'].to(dev)
                label = b['label'].to(dev)

                pred = model(ids)
                loss = crit(pred, label)
                acc = self.accuracy_check(pred, label)

                accs.append(acc.item())
                losses.append(loss.item())
        
        return np.mean(accs), np.mean(losses)