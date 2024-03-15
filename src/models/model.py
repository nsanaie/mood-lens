import torch

class Model(torch.nn.Module):

    def __init__(self, model, output, freeze):
        super().__init__()
        self.model = model
        hidden = model.config.hidden_size

        self.fc = torch.nn.Linear(hidden, output)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, ids):
        output = self.model(ids, output_attentions=True)
        hidden = output.last_hidden_state
        attention_mask = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        return self.fc(torch.tanh(cls_hidden))