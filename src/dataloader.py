from imports import *

class CustomDataset:

    def __init__(self, config: dict, root: str = ".", wandb=None):
        self.root = root
        self.config = config
        self.wandb = wandb

    def load(self):
        
        config = self.config

        if config["dataset_type"]=="remote":
            dataset = load_dataset(config["dataset"], split="train[:50]")
            dataset = dataset.rename_column("context", "text") #for squad
        
        elif config["dataset_type"]=="txt":    
            text = []
            for path in config["dataset"]:
                with open(path, "r") as f:
                    text+=[x.strip() for x in f.readlines()]
            dataset = Dataset.from_dict({"text":text})

        elif config["dataset_type"]=="csv":
            dataset = load_dataset("csv", data_files=config["dataset"], split="train")
            dataset = dataset.rename_column("context", "text")
        
        return dataset