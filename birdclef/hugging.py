from huggingface_hub import notebook_login
from datasets import load_dataset, Audio
import tensorflow as tf
from transformers import AutoFeatureExtractor
import evaluate
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np


# TODO: UNTESTED! Test after finished training
class HuggingModel:
    def __init__(self, input_shape, labels):
        self.labels = labels
        self.num_labels = len(labels)
        self.input_shape = input_shape

        self.label2id, self.id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            self.label2id[label] = str(i)
            self.id2label[str(i)] = label

        notebook_login()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("saved_model/1")
        self.accuracy = evaluate.load("accuracy")

        self.model = AutoModelForAudioClassification.from_pretrained(
            "saved_model/1",
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.model.summary()

    def preprocess_function(self, examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.feature_extractor(
            audio_arrays, sampling_rate=self.feature_extractor.sampling_rate, max_length=16000, truncation=True
        )
        return inputs

    def compute_metrics(self, eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    def train(self, data, val_data, epochs=100, checkpoint_filepath='saved_model/1', batch_size=30, verbose=1):
        training_args = TrainingArguments(
            output_dir="my_awesome_mind_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data,
            eval_dataset=val_data,
            tokenizer=self.feature_extractor,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        # trainer.push_to_hub()

