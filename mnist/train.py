"""
Trains a Pytorch image classification model using device-agnostic code.
"""
import os
import torch
from mnist import data_setup, engine, model_builder, utils
from torchvision.transforms import ToTensor

# Hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders("data", ToTensor, BATCH_SIZE)

torch.manual_seed(13)
model = model_builder.TinyVGG(
    input_shape=1,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

engine.train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    epochs=NUM_EPOCHS,
    device=device
)

utils.save_model(model, target_dir='models', model_name='modular_tinyvgg_adam.pth')
