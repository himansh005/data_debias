from imports import *

class CustomDataCollator(DataCollatorForLanguageModeling):

  def __init__(self, tokenizer, mlm = True, mlm_probability=None, pad_to_multiple_of= None, tf_experimental_compile = False, return_tensors = "pt"):
    super().__init__(tokenizer, mlm = True, mlm_probability = mlm_probability, pad_to_multiple_of= None, tf_experimental_compile = False, return_tensors = "pt")
  
  def torch_mask_tokens(self, inputs, special_tokens_mask):
    
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    replace_tokens = CustomTrainer.replace_tokens

    ignore_indices = torch.isin(inputs,torch.Tensor(replace_tokens))
    labels[~masked_indices] = -100  
    labels[~ignore_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & (masked_indices & ignore_indices )
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    
    inputs[indices_random] = random_words[indices_random]
    
    for l in labels:
      words = self.tokenizer.decode(l[l!=-100])
      if len(words)>0:
        CustomTrainer.mlm_words+=words.split(" ")

    return inputs, labels

class CustomTrainer:

    mlm_words = []
    replace_tokens = []

    def __init__(self, config: dict, root: str = ".", wandb=None):
        self.config = config
        self.root = root
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.wandb = wandb

    def _group_texts(self, examples):
        # Concatenate all texts.
        max_length = self.max_length
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def train(self, dataset, words):
        
        dataset = dataset.remove_columns("bias")

        d = dataset.train_test_split(test_size=0.1)
        def dataset_to_text(dataset, output_filename="data.txt"):
            """Utility function to save dataset text to disk,
            useful for using the texts to train the tokenizer 
            (as the tokenizer accepts files)"""
            with open(os.path.join(tempfile.gettempdir(), output_filename), "w") as f:
                for t in dataset["text"]:
                    print(t, file=f)

        dataset_to_text(d["train"], "train.txt")
        dataset_to_text(d["test"], "test.txt")

        # special_tokens = [
        #     "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
        # ]
        
        # files = ["train.txt"]
        vocab_size = 30_522
        max_length = self.config["max_length"]
        self.max_length = max_length
        truncate_longer_samples = False
        
        # # initialize the WordPiece tokenizer
        #tokenizer = BertWordPieceTokenizer.from_pretrained("bert-base-uncased")
        # # train the tokenizer
        #tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
        # # enable truncation up to the maximum 512 tokens
        #tokenizer.enable_truncation(max_length=max_length)

        model_path = "pretrained"
        if not os.path.isdir(model_path):
            os.mkdir(os.path.join(tempfile.gettempdir(), model_path))

        # # save the tokenizer  
        #tokenizer.save_model(model_path)

        # # dumping some of the tokenizer config to config file, 
        # # including special tokens, whether to lower case and the maximum sequence length
        # with open(os.path.join(model_path, "config.json"), "w") as f:
        #   tokenizer_cfg = {
        #       "do_lower_case": True,
        #       "unk_token": "[UNK]",
        #       "prompt_token": "[PMT]",
        #       "sep_token": "[SEP]",
        #       "pad_token": "[PAD]",
        #       "cls_token": "[CLS]",
        #       "mask_token": "[MASK]",
        #       "model_max_length": max_length,
        #       "max_len": max_length,
        #   }
        #   json.dump(tokenizer_cfg, f)

        # when the tokenizer is trained and configured, load it as BertTokenizerFast
        
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        #special_tokens_dict = {'additional_special_tokens': ['[START]', '[STOP]']}
        
        #num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        # check if the tokens are already in the vocabulary
        new_tokens = set(words) - set(tokenizer.vocab.keys())

        # add the tokens to the tokenizer vocabulary
        tokenizer.add_tokens(list(new_tokens))
        CustomTrainer.replace_tokens = tokenizer.encode(" ".join(list(new_tokens)))

        def encode_with_truncation(examples):
            """Mapping function to tokenize the sentences passed with truncation"""
            return tokenizer(examples["text"], truncation=True, padding="max_length",
                            max_length=max_length, return_special_tokens_mask=True)

        def encode_without_truncation(examples):
            """Mapping function to tokenize the sentences passed without truncation"""
            return tokenizer(examples["text"], return_special_tokens_mask=True)

        encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
        train_dataset = d["train"].map(encode, batched=True)
        test_dataset = d["test"].map(encode, batched=True)

        if truncate_longer_samples:
            train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
            test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        else:
            test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
            train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
       

        if not truncate_longer_samples:
            train_dataset = train_dataset.map(self._group_texts, batched=True,
                                            desc=f"Grouping texts in chunks of {max_length}")
            test_dataset = test_dataset.map(self._group_texts, batched=True,
                                            desc=f"Grouping texts in chunks of {max_length}")
            # convert them from lists to torch tensors
            train_dataset.set_format("torch")
            test_dataset.set_format("torch")

        # initialize the model with the config
        model_config = AutoConfig.from_pretrained(self.config["model"])#(self.config["model"], vocab_size=vocab_size, max_position_embeddings=max_length)
        model = AutoModelForMaskedLM.from_pretrained(self.config["model"], config=model_config)
        model.resize_token_embeddings(len(tokenizer))
        
        # for name, param in model.named_parameters():
        #    print(name, param.requires_grad)

        # sys.exit(0)

        data_collator = CustomDataCollator(
            tokenizer=tokenizer, mlm=True, mlm_probability=self.config["mlm_prob"]
        )
        
        
        training_args = TrainingArguments(
            output_dir=tempfile.gettempdir(),          # output directory to where save model checkpoint
            evaluation_strategy="epoch",    # evaluate each `logging_steps` steps
            overwrite_output_dir=True,      
            num_train_epochs=self.config["epochs"],            # number of training epochs, feel free to tweak
            per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
            gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
            per_device_eval_batch_size=64,  # evaluation batch size
            logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
            save_steps=1000,
            load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
            save_total_limit=1,           # whether you don't have much space so you let only 3 model weights saved in the disk,
            save_strategy = "epoch",
            seed=self.config["seed"],
            report_to="wandb"

        )

        # initialize the trainer and pass everything to it
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator
        )

        # train the model
        gc.collect()
        torch.cuda.empty_cache()
        trainer.train()
        
        df = pd.DataFrame.from_records(list(dict(Counter(CustomTrainer.mlm_words)).items()), columns=['word','count']).sort_values(by="count")
        df = wandb.Table(dataframe=df)
        self.wandb.log({"mlm words":df})

        self.logger.debug("MLM words: {}".format(CustomTrainer.mlm_words))
        model_path = os.path.join(self.root, self.config["persistence_dir"], "models", str(self.config["experiment_id"]), self.config["model"])
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)

        return model_path