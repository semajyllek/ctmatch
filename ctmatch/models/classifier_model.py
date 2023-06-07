
import logging
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime import ORTOptimizer
import evaluate

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import torch

from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from nn_pruning.inference_model_patcher import optimize_model
from nn_pruning.sparse_trainer import SparseTrainer


from ..pipeconfig import PipeConfig
from ..dataprep import DataPrep


logger = logging.getLogger(__name__)

PRUNED_HUB_MODEL_NAME = 'semaj83/scibert_finetuned_pruned_ctmatch'


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




class PruningTrainer(SparseTrainer, WeightedLossTrainer):
    def __init__(self, sparse_args, *args, **kwargs):
        WeightedLossTrainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)


class ClassifierModel:
    
    def __init__(self, model_config: PipeConfig, data: DataPrep, device: str):
        self.model_config = model_config
        self.dataset = data.ct_dataset
        self.tokenizer = data.classifier_tokenizer
        self.tokenize_func = data.tokenize_function
        self.trainer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = device

        if not self.model_config.ir_setup:
            self.train_dataset_df = data.ct_dataset['train'].to_pandas()
            self.num_training_steps = self.model_config.train_epochs * len(self.dataset['train'])

        self.model = self.load_model()
        self.pruned_model = None
    
        if not self.model_config.use_trainer and not self.model_config.ir_setup:
            self.train_dataloader, self.val_dataloader = self.get_dataloaders()

        
        if self.model_config.prune:
             self.prune_trainer = None
             self.sparse_args = self.get_sparse_args()
             self.mpc = self.get_model_patching_coordinator()
               

    # ------------------ Model Loading ------------------ #
    def get_model(self):
        if self.model_config.num_classes == 0:
            return AutoModelForSequenceClassification.from_pretrained(self.model_config.classifier_model_checkpoint)

        id2label, label2id = self.get_label_mapping()
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.classifier_model_checkpoint,
            num_labels=self.model_config.num_classes,     # makes the last head be replaced with a linear layer with num_labels outputs (fine-tuning)
            id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True                  # because of pruned model changes
        )

        return self.add_pad_token(model)


    def add_pad_token(self, model):
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        return model


    def load_model(self):
        self.model = self.get_model()

        if self.model_config.ir_setup:
            return self.model
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.model_config.learning_rate, weight_decay=self.model_config.weight_decay)
        self.num_training_steps = self.model_config.train_epochs * len(self.dataset['train'])
        self.lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=self.optimizer, 
            num_warmup_steps=self.model_config.warmup_steps, 
            num_training_steps=self.num_training_steps
        )

        if self.model_config.use_trainer and not self.model_config.prune:
            self.trainer = self.get_trainer()
        else:
            self.model = self.model.to(self.device)

        return self.model


    def get_label_mapping(self):
        #id2label = {idx:self.dataset['train'].features["labels"].int2str(idx) for idx in range(3)}
        id2label =  {'0':'not_relevant', '1':'partially_relevant', '2':'relevant'}
        label2id = {v:k for k, v in id2label.items()}
        return id2label, label2id

    def get_label_weights(self):
        label_weights = (1 - (self.train_dataset_df["labels"].value_counts().sort_index() / len(self.train_dataset_df))).values
        label_weights = torch.from_numpy(label_weights).float().to("cuda")


    def get_trainer(self):
        return WeightedLossTrainer(
            model=self.model,
            optimizers=(self.optimizer, self.lr_scheduler),
            args=self.get_training_args_obj(),
            compute_metrics=self.compute_metrics,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            label_weights=self.get_label_weights()
        )


    def get_training_args_obj(self):
        output_dir = self.model_config.output_dir if self.model_config.output_dir is not None else self.model_config.classifier_data_path.parent.parent.as_posix()
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.model_config.train_epochs,
            learning_rate=self.model_config.learning_rate,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            weight_decay=self.model_config.weight_decay,
            evaluation_strategy="epoch",
            logging_steps=len(self.dataset["train"]) // self.model_config.batch_size,
            fp16=self.model_config.fp16
        )
        
    

    def train_and_predict(self):
        if self.trainer is not None:
            self.trainer.train()
            predictions = self.trainer.predict(self.dataset["test"])
            logger.info(predictions.metrics.items())
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=self.get_label_weights())
            self.manual_train()
            self.manual_eval()



     # ------------------ native torch training loop ------------------ #
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(self.dataset['train'], shuffle=True, batch_size=self.model_config.batch_size)
        val_dataloader = DataLoader(self.dataset['validation'], batch_size=self.model_config.batch_size)
        return train_dataloader, val_dataloader



    # taken from ctmatch for messing about 
    def manual_train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train()
        for epoch in range(self.model_config.train_epochs):
            for batch in tqdm(self.train_dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = self.loss_func(outputs.logits, batch['labels'])
                #total_loss += loss.item()
                loss.backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.manual_eval()
                logger.info(f"{loss=}")
                progress_bar.update(1)
            



    def manual_eval(self):
        metric = evaluate.load("f1")
        self.model.eval()
        for batch in self.val_dataloader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
        
            # don't learn during evaluation
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        
        logger.info(metric.compute(average='weighted'))




    def get_sklearn_metrics(self):
        with torch.no_grad():
            if self.model_config.use_trainer:
                if self.model_config.prune:
                    self.prune_trainer.model.to(self.device)
                    logger.info("using pruned trainer model")
                    preds = self.prune_trainer.predict(self.dataset['test']).predictions
                else:    
                    preds = self.trainer.predict(self.dataset['test']).predictions

                if "bart" in self.model_config.name:
                    preds = preds[0]

                y_preds = list(preds.argmax(axis=1))
            else:

                if self.model_config.prune:
                    model = self.pruned_model.to(self.device)
                else:
                    model = self.model.to(self.device)
                y_preds = []
                for input_ids in self.dataset['test']['input_ids']:
                    input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
                    y_pred = model(input_ids).logits.argmax().item()
                    y_preds.append(y_pred)
         
        y_trues = list(self.dataset['test']['labels'])
        return confusion_matrix(y_trues, y_preds), classification_report(y_trues, y_preds)
    

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions
        if "bart" in self.model_config.name:
            preds = preds[0]
        
        preds = preds.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        return {"f1":f1}
    
    def inference_single_example(self, topic: str, doc: str, return_preds: bool = False) -> str:
        """
        desc: method to predict relevance label on new topic, doc examples 
        """
        ex = {'doc':doc, 'topic':topic}
        with torch.no_grad():
            inputs = torch.LongTensor(self.tokenize_func(ex)['input_ids']).unsqueeze(0)
            outputs = self.model(inputs).logits
            if return_preds:
                return torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
            return str(outputs.argmax().item())


    def batch_inference(self, topic: str, docs: List[str], return_preds: bool = False) -> List[str]:
        topic_repeats = [topic for _ in range(len(docs))]
        inputs = self.tokenizer(
            topic_repeats, docs, return_tensors='pt', 
            truncation=self.model_config.truncation, 
            padding=self.model_config.padding, 
            max_length=self.model_config.max_length
        )

        with torch.no_grad():
            outputs = torch.nn.functional.softmax(self.model(**inputs).logits, dim=1)
        
        if return_preds:
            return outputs
        
        return outputs.argmax(dim=1).tolist()



    # ------------------ pruning  ------------------ #

    def prune_model(self):
        self.mpc.patch_model(self.model)
        self.model.save_pretrained("models/patched")
        self.prune_trainer = self.get_pruning_trainer()
        self.prune_trainer.set_patch_coordinator(self.mpc)
        self.prune_trainer.train()
        self.mpc.compile_model(self.prune_trainer.model)
        self.pruned_model = optimize_model(self.prune_trainer.model, "dense")
        if self.model_config.push_to_hub:
            self.pruned_model.push_to_hub(PRUNED_HUB_MODEL_NAME)


    def get_sparse_args(self):
        sparse_args = SparseTrainingArguments()

        hyperparams = {
            "dense_pruning_method": "topK:1d_alt", 
            "attention_pruning_method": "topK", 
            "initial_threshold": 1.0, 
            "final_threshold": 0.5, 
            "initial_warmup": 1,
            "final_warmup": 3,
            "attention_block_rows":32,
            "attention_block_cols":32,
            "attention_output_with_dense": 0
        }

        for k,v in hyperparams.items():
            if hasattr(sparse_args, k):
                setattr(sparse_args, k, v)
            else:
                print(f"sparse_args does not have argument {k}")

        return sparse_args

    
    def get_pruning_trainer(self):
        return PruningTrainer(
            sparse_args=self.sparse_args,
            args=self.get_training_args_obj(),
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            label_weights=self.get_label_weights()
        )



    def get_model_patching_coordinator(self):
        return ModelPatchingCoordinator(
            sparse_args=self.sparse_args, 
            device=self.device, 
            cache_dir="checkpoints", 
            logit_names="logits", 
            teacher_constructor=None
        )
    
    
    # onyx optimization
    def optimize_model(self):  
        onnx_path = Path("onnx")
        model_id = self.model_config.classifier_model_checkpoint
        #assert self.pruned_model is not None, "pruned model must be loaded before optimizing"
        opt_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        optimizer = ORTOptimizer.from_pretrained(opt_model)
        optimization_config = OptimizationConfig(optimization_level=99) # enable all optimizations
        optimizer.optimize(
            save_dir=onnx_path,
            optimization_config=optimization_config,
        )
        opt_model.save_pretrained(onnx_path)
        self.tokenizer.save_pretrained(onnx_path)  

        #optimized_model = ORTModelForSequenceClassification.from_pretrained(onnx_path, file_name="model_optimized.onnx")  

        return opt_model

       



        
