import torch
import datasets
# import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer


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

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
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


if __name__ == "__main__":
    data_model = ColaDataset()
    data_model.prepare_data()
    data_model.setup()
    # print(data_model.train_data[0])
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)