# write a function to test transformers.get_cosine_schedule_with_warmup

from transformers import get_cosine_schedule_with_warmup
import torch
import matplotlib.pyplot as plt 

num_epochs = 100
num_updates = 10
def train_epoch(scheduler, optimizer):
    for i in range(num_updates):
        optimizer.step()
        scheduler.step()

model = torch.nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=num_epochs)
lr_list = []
for i in range(num_epochs):
    train_epoch(scheduler, optimizer)
    lr_list.append(scheduler.get_last_lr()[0])
    
plt.plot(lr_list)
plt.show()