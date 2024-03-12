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

class Modeler:
    """
    Wrapper for creating, training, saving and other handling of the sentiment model.
    """
    def __init__(self, training):

        if training:
            self.model_name = "bert-base-uncased"
        else:
            self.model_name = ""

        #output directory
        self.outputp = "model"

        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # setting up model with configs, device to run on
        self.config = AutoConfig(self.model_name)
        self.model = Model.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # evaluation mode on initilization of modeler
        self.model.eval()
    
    def train(self, train_loader, optimizer, criterion):
        # set to training mode
        self.model.train()
        # create batches (tqdm is loading bar plugin)
        for inputs_ids, attention_mask, label in tqdm(train_loader):

            # add data to device (including labels)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # gradient reset
            optimizer.zero_grad()

            # get output and loss + backpropogation of loss
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            loss = criterion(outputs, labels.float())
            loss.backward()

            # omptimze? the mode. Need to look more into this.
            optimizer.step()


            