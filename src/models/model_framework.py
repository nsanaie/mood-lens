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
    Creating, training, saving and other handling of the sentiment model.
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

        # evaluation mode
        self.model.eval()
