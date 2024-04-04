
import os
import torch 
from transformers import DataCollatorWithPadding, AutoTokenizer

from data import ColaDataset
from model import ColaModel, FinetunedLLM

from transformers import Trainer, TrainerCallback, TrainingArguments
from sklearn.metrics import accuracy_score

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
            
  
  
epochs = 2
batch_size = 4
lr = 0.001

model_name = "google/bert_uncased_L-2_H-128_A-2"
cola_dataset = ColaDataset(batch_size=batch_size)
train_loader = cola_dataset.train_dataloader()
val_loader = cola_dataset.val_dataloader()



model = FinetunedLLM()
tokenizer = AutoTokenizer.from_pretrained(model_name)
# callback = PrinterCallback
# training_args = TrainingArguments("test_train", num_train_epochs=3, logging_dir="./logging")
# data_collator = DataCollatorWithPadding(tokenizer  = tokenizer)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
citerion = torch.nn.CrossEntropyLoss()

checkpoint_dir = './checkpoint/best_model.pt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

history = dict(
    loss  = [],
    val_loss = [], 
    acc  = [],
    val_acc = []
)
best_loss_val = torch.inf
for epoch in range(epochs):
    print("--------Start Training ------>")
    model.train()
    total_loss = 0
    total_acc = 0 
    total_val_loss = 0 
    total_val_acc = 0
    for idx, train_batch in enumerate(train_loader):
        print(f"Batch Index: {idx}")
        input_ids = train_batch['input_ids'].to(device)
        attention_mask = train_batch['attention_mask'].to(device)
        label = train_batch['label'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = citerion(output, label)
        _, preds = torch.max(output,dim=1)
        acc = accuracy_score(preds.cpu(),label.cpu())
        acc = torch.tensor(acc)

        history['loss'].append(loss.item())
        history['acc'].append(acc)
        total_loss += loss
        total_acc += acc 
        if idx %10 == 0: 
            if best_loss_val > total_loss/len(train_loader):
                best_loss_val = total_loss/len(val_loader)
                torch.save(model, checkpoint_dir)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch} | Loss: {total_loss/len(train_loader)}| Acc: {total_acc/len(train_loader)}")
    model.eval()
    for idx, val_batch in enumerate(val_loader):
        input_ids = val_batch['input_ids'].to(device)
        attention_mask = val_batch['attention_mask'].to(device)
        label = val_batch['label'].to(device)
        #eval model
        out = model(input_ids, attention_mask)
        val_loss = citerion(out, label)
        _, preds = torch.max(out,dim=1)
        val_acc = accuracy_score(preds.cpu(),label.cpu())
        val_acc = torch.tensor(val_acc)
        
        #store value
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc)
        
        total_val_loss += val_loss 
        total_val_acc += val_acc
    if best_loss_val > total_val_loss/len(val_loader):
        best_loss_val = total_val_loss/len(val_loader)
        torch.save(model, checkpoint_dir)

    print(f"Epoch: {epoch} | Loss: {total_loss/len(train_loader)}| Acc: {total_acc/len(train_loader)}| Val_loss: {total_val_loss/len(val_loader)}| Val_acc: {total_val_acc/len(val_loader)}")
    
    
    
    import pickle 
    
    with open("history.pkl", 'wb') as f:
        pickle.dump(file=f)
    
    

