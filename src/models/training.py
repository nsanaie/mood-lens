from tqdm import trange
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model_framework import Modeler
from dataset import TextAndLabelsSet


if __name__ == "__main__":

    # using DistilBERT model (faster and more lightweight than bert? Can change)
    model_name = "bert-base-uncased"

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create modeler
    model_wrapper = Modeler(model_name=model_name, tokenizer=tokenizer)

    # create dataset + dataloaders for training and evaluation
    training = TextAndLabelsSet(path="data/train.tsv", tokenizer=tokenizer)
    training_loader = DataLoader(dataset=training, batch_size=32, num_workers=4)
    valditation = TextAndLabelsSet(path="data/test.tsv", tokenizer=tokenizer)
    valditation_loader = DataLoader(dataset=valditation, batch_size=32, num_workers=4)

    # create criterion and optimizer to use for training
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model_wrapper.model.parameters(), lr=2e-5)

    # keep track of best training epoch
    accuracy = 0
    epochs = 4

    # epoch iteration
    for epoch in trange(epochs):

        # train modeler warpper for each epoch
        new_accuracy = model_wrapper.train_model(train_loader=training_loader, val_loader=valditation_loader, optimizer=optimizer, criterion=criterion, epoch_count=epoch)

        if new_accuracy > accuracy:
            accuracy = new_accuracy
            model_wrapper.save_model()
        