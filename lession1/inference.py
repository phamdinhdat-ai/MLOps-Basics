import torch 

import torch.nn.functional as F
from model import ColaModel
from data import ColaDataset






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
    
    
batch_idx = {'idx': 0, 'label': 1, 'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}
text = "Our friends won't buy this analysis, let alone the next one we propose."

model = ColaPreictor(checkpoint_dir='./checkpoint/best_model.pt')
out_idx, labels  = model.predict(batch_idx=batch_idx)
print(f"True answer: {batch_idx['label']}| Predict Answer: {labels}")
