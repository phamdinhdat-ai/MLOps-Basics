# Lession 1: Project setup 

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

Refer to the [Blog Post here](https://www.ravirajag.dev/blog/mlops-project-setup-part1)

The project I have implemented is a simple classification problem. The scope of this week is to understand the following topics:

- `How to get the data?`
- `How to process the data?`
- `How to define dataloaders?`
- `How to declare the model?`
- `How to train the model?`
- `How to do the inference?`


What things we should use:

- [Huggingface Datasets](https://github.com/huggingface/datasets)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/)

## Dataset

I will be using CoLA(Corpus of Linguistic Acceptability) dataset. The task is about given a sentence it has to be classified into one of the two classes.

- ❌ Unacceptable: Grammatically not correct

- ✅ Acceptable: Grammatically correct

I am using (Huggingface datasets) to download and load the data. It supports 800+ datasets and also can be used with custom datasets.

Downloading the dataset is as easy as

``` python 
cola_dataset = load_dataset("glue", "cola")
print(cola_dataset)
```

``` python 
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 8551
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1043
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1063
    })
})
```

Let's see the sample datapoint: 

```python
train_dataset = cola_dataset['train']
print(train_dataset[0])
```

Output:
``` python
{
    'idx': 0,
    'label': 1,
    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."
}
```

## Prepare Data
Get data and format data as input of LLM model. I used `BertTokenizer` to tokenize text in the dataset.

``` python 
class ColaDataset(Dataset):
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", batch_size=8):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(model_name,return_dict = False)
        self.prepare_data()
        self.setup()
        
    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]
        
    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def setup(self):
        # we set up only relevant datasets when stage is specified
        self.train_data = self.train_data.map(self.tokenize_data, batched=True)
        self.train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        self.val_data = self.val_data.map(self.tokenize_data, batched=True)
        self.val_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )

```

## Build Fintune LM model

It's easy to create and fintune Language model by using [Huggingface Transformers](https://github.com/huggingface/transformers). 
``` python 

class FinetunedLLM(nn.Module):  # pragma: no cover, torch model
    """Model architecture for a Large Language Model (LLM) that we will fine-tune."""

    def __init__(self, model_name="allenai/scibert_scivocab_uncased", dropout_p=0.4, num_classes = 2):
        super(FinetunedLLM, self).__init__()
        self.llm = BertModel.from_pretrained(model_name, return_dict = False)
        self.hidden_size = self.llm.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        seq, pool = self.llm(input_ids=input_ids , attention_mask=attention_mask)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z

``` 

## Train model 

I trained model from scratch, but you also can use `Trainer` and `TrainerArguments` to set your own parameters. 

``` python 
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
    
   
``` 

## Inference model

``` python 
    class ColaPreictor: 
    def __init__(self, checkpoint_dir):
        self.ckpoint = checkpoint_dir
        self.model  = torch.load(checkpoint_dir)
        self.model.eval()
        self.tokenizer = ColaDataset()
        self.labels = ["unacceptable", "acceptable"]
        
    def predict(self, batch_idx):
        
        encoded = self.tokenizer.tokenize_data(batch_idx)
        input_ids = torch.tensor([encoded['input_ids']])
        attention_mask = torch.tensor([encoded['attention_mask']])
        logits = self.model(input_ids, attention_mask)
        out = F.softmax(logits, dim=1)
        output_idx = torch.argmax(out, dim = 1)
        return output_idx, self.labels[output_idx]
    
class_names = ["unacceptable", "acceptable"]
batch_idx = {'idx': 0, 'label': 1, 'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}
text = "Our friends won't buy this analysis, let alone the next one we propose."

model = ColaPreictor(checkpoint_dir='./checkpoint/best_model.pt')
out_idx, labels  = model.predict(batch_idx=batch_idx)
print(f"True answer: {class_names[batch_idx['label']]}| Predict Answer: {labels}")

```

Output: 
``` 
True answer: acceptable | Predict Answer: unacceptable
``` 