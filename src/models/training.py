from tqdm import trange
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os

from model_framework import Modeler
from dataset import TextAndLabelsSet


if __name__ == "__main__":

    # using DistilBERT model (faster and more lightweight than bert? Can change)
    model_name = "distilbert-base-uncased"

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create modeler
    modeler = Modeler(model_name=model_name, tokenizer=tokenizer)

    # create dataset + dataloaders for training and evaluation
    training = TextAndLabelsSet(path=os.path.abspath("data/train_binary_sent.csv"), tokenizer=tokenizer)
    training_loader = DataLoader(dataset=training, batch_size=64, num_workers=4)
    valditation = TextAndLabelsSet(path="data/test_binary_sent.csv", tokenizer=tokenizer)
    valditation_loader = DataLoader(dataset=valditation, batch_size=64, num_workers=4)

    # create criterion and optimizer to use for training
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=modeler.model.parameters(), lr=2e-5)

    # keep track of best training epoch
    accuracy = 0
    epochs = 4

    # epoch iteration
    for epoch in trange(epochs):

        # train modeler warpper for each epoch
        new_accuracy = modeler.train_model(train_loader=training_loader, val_loader=valditation_loader, optimizer=optimizer, criterion=criterion, epoch_count=epoch)

        if new_accuracy > accuracy:
            accuracy = new_accuracy
            modeler.save_model()
        