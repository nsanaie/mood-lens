import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoConfig, DistilBertPreTrainedModel, DistilBertModel
from tqdm import tqdm

class Model(DistilBertPreTrainedModel):
    """
    Creating the model to be trained from the original DistilBERT
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = DistilBertModel(config)
        self.linear_layer = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        return self.linear_layer(last_hidden_state[:, 0])


class Modeler:
    """
    Wrapper for creating, training, saving and other handling of the sentiment model.
    """
    def __init__(self, model_name, tokenizer):

        self.model_name = model_name
        self.tokenizer = tokenizer

        # setting up model with configs, device to run on
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = Model.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # evaluation mode on initilization of modeler
        self.model.eval()
    
    def train_model(self, train_loader, val_loader, optimizer, criterion, epoch_count):
        """
        One epoch training for use in training.py
        """
        # set to training mode
        self.model.train()
        # create batches (tqdm is loading bar plugin)
        for input_ids, attention_mask, labels in tqdm(train_loader, desc="training"):

            # add data to device (including labels)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # gradient reset
            optimizer.zero_grad()

            # get output and loss + backpropogation of loss
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
            loss = criterion(input=outputs, target=labels.float())
            loss.backward()

            # omptimze? the mode. Need to look more into this.
            optimizer.step()
        
        # set to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            
            val_accuracy, val_loss = 0, 0
            i = 0
            # go through in batches for evaluation
            for val_input_ids, val_attention_mask, val_labels in tqdm(val_loader, desc="evaluating"):
            
                # add data to device (including labels)
                val_input_ids = val_input_ids.to(self.device)
                val_attention_mask = val_attention_mask.to(self.device)
                val_labels = val_labels.to(self.device)

                val_logits = self.model(input_ids=val_input_ids, attention_mask=val_attention_mask)
                val_loss += criterion(input=val_logits.squeeze(-1), target=val_labels.float()).item()
                
                predictions = torch.sigmoid(val_logits.unsqueeze(-1))
                binary_predictions = (predictions > 0.5).long().squeeze()
                val_accuracy += (binary_predictions == val_labels).float().mean()

                i += 1


        accuracy = val_accuracy / i
        print(f'Epoch {epoch_count} Completed. Training Loss: {loss.item()}, Validation Loss: {val_loss}, Accuracy: {accuracy.item()}')
        return accuracy.item()


        
    def save_model(self):
        """
        Save model, tokenizer and configuration
        """
        self.model.save_pretrained(save_directory="DistilBERT/")
        self.config.save_pretrained(save_directory="DistilBERT/")
        self.tokenizer.save_pretrained(save_directory="DistilBERT/")
