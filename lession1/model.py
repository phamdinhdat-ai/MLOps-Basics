import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score

from transformers import BertModel












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




class ColaModel(nn.Module):
    def __init__(self, model_name ="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        
        self.bert = BertForSequenceClassification.from_pretrained(model_name)
        self.fc = nn.Linear(in_features= self.bert.config.hidden_size, out_features=2)
        self.num_classes = 2
        self.bert.eval()
    def forward(self, input_ids , attention_mask):
        out_bert = self.bert(input_ids= input_ids, attention_mask = attention_mask)
        

        logits = self.fc(out_bert)
        return logits
    
    def training_step(self, batch , batch_idx):
        logits = self.forward(batch['input_ids'], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        # self.log("train_loss", loss, prog_bar = True)
        
        return loss 
    
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        # self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        val_acc = torch.tensor(val_acc)
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])