
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch import nn
import evaluate
import torch

from ..utils.ctmatch_utils import compute_metrics
from ..modelconfig import ModelConfig
from ..dataprep import DataPrep



class WeightedLossTrainer(Trainer):

  def __init__(self, label_weights, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.label_weights = label_weights

  def compute_loss(self, model, inputs, return_outputs=False):
    outputs = model(**inputs)
    logits = outputs.get("logits")
    labels = inputs.get("labels")
    loss_func = nn.CrossEntropyLoss(weight=self.label_weights)
    loss = loss_func(logits, labels)
    return (loss, outputs) if return_outputs else loss



class LModel:
    
    def __init__(self, model_config: ModelConfig, data: DataPrep):
        self.model_config = model_config
        self.dataset = data.ct_dataset
        self.train_dataset_df = data.ct_dataset['train'].to_pandas()
        self.tokenizer = data.tokenizer
        self.trainer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.num_training_steps = self.model_config.train_epochs * len(self.dataset['train'])
        self.model = self.load_model()
    
        if not self.model_config.use_trainer:
            self.train_dataloader, self.val_dataloader = self.get_dataloaders()

               

    # ------------------ Model Loading ------------------ #
    def get_model(self):
        if self.model_config.num_classes == 0:
            return AutoModelForSequenceClassification.from_pretrained(self.model_config.model_checkpoint)
        
        if self.model_config.model_checkpoint == 'microsoft/biogpt':
            return AutoModelForCausalLM.from_pretrained(self.model_config.model_checkpoint)
        
        id2label, label2id = self.get_label_mapping()
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_checkpoint,
            num_labels=self.model_config.num_classes,     # makes the last head be replaced with a linear layer with num_labels outputs (fine-tuning)
            id2label=id2label, label2id=label2id
        )
            



    def load_model(self):
        self.model = self.get_model()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
       
        self.optimizer = AdamW(self.lm_model.parameters(), lr=self.model_config.learning_rate, weight_decay=self.model_config.weight_decay)
        self.num_training_steps = self.model_config.train_epochs * len(self.dataset['train'])
        self.lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=self.optimizer, 
            num_warmup_steps=self.model_config.warmup_steps, 
            num_training_steps=self.num_training_steps
        )

        if self.model_config.use_trainer:
            self.trainer = self.get_trainer()
        else:
            self.model = self.model.to(self.device)

        return self.model


    def get_label_mapping(self):
        id2label = {idx:self.dataset['train'].features["labels"].int2str(idx) for idx in range(3)}
        label2id = {v:k for k, v in id2label.items()}
        return id2label, label2id

    def get_label_weights(self):
        label_weights = (1 - (self.train_dataset_df["labels"].value_counts().sort_index() / len(self.train_dataset_df))).values
        label_weights = torch.from_numpy(label_weights).float().to("cuda")


    def get_trainer(self):
        return WeightedLossTrainer(
            model=self.lm_model,
            optimizers=(self.optimizer, self.lr_scheduler),
            args=self.get_training_args_obj(),
            compute_metrics=compute_metrics,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            label_weights=self.get_label_weights()
        )


    def get_training_args_obj(self):
        output_dir = self.model_config.output_dir if self.model_config.output_dir is not None else self.model_config.data_path.parent.parent.as_posix()
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.model_config.train_epochs,
            learning_rate=self.model_config.learning_rate,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            weight_decay=self.model_config.weight_decay,
            evaluation_strategy="epoch",
            logging_steps=len(self.dataset["train"]) // self.model_config.batch_size,
            push_to_hub=self.model_config.push_to_hub
        )
        
    

    def train_and_predict(self):
        if self.trainer is not None:
            self.trainer.train()
            predictions = self.trainer.predict(self.dataset["test"])
            print(predictions.metrics.items())
        else:
            self.torch_train()
            self.torch_eval()



     # ------------------ native torch training loop ------------------ #
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(self.dataset['train'], shuffle=True, batch_size=self.model_config.batch_size)
        val_dataloader = DataLoader(self.
        dataset['validation'], batch_size=self.model_config.batch_size)
        return train_dataloader, val_dataloader



    def torch_train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        self.lm_model.train()
        for epoch in range(self.model_config.train_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.lm_model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.torch_eval()
                progress_bar.update(1)
            

    def torch_eval(self):
        metric = evaluate.load("f1")
        self.lm_model.eval()
        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # don't learn during evaluation
            with torch.no_grad():
                outputs = self.lm_model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        print(metric.compute(average='weighted'))




    def get_sklearn_metrics(self):
        if self.model_config.use_trainer:
            y_preds = list(self.trainer.predict(self.dataset["validation"]).predictions.argmax(axis=1))
        else:
            y_preds = []
            for input_ids in self.dataset['validation']['input_ids']:
                y_pred = self.lm_model(input_ids).logits.argmax().item()
                y_preds.append(y_pred)
         
        y_trues = list(self.dataset["validation"]["labels"])
        return confusion_matrix(y_trues, y_preds), classification_report(y_trues, y_preds)
    
