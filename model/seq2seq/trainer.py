from tqdm import tqdm
import collections
from typing import Dict, List, Optional, NamedTuple
import transformers.trainer_seq2seq
from datasets.arrow_dataset import Dataset
from transformers.trainer_utils import PredictionOutput, speed_metrics, EvalLoopOutput
from datasets.metric import Metric
from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor

# from model.utils.beam import RABeamScorer
import numpy as np
import torch
import time
import json

import numpy as np

def softmax(x, axis=-1):
    max = np.max(x,axis=axis,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=axis,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]

class T5SpiderTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
        self,
        metric: Metric,
        *args,
        eval_examples: Optional[Dataset] = None, 
        ignore_pad_token_for_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.beam_scorer = BeamSearchScorer(
                    batch_size=1,
                    num_beams=8,
                    device=self.model.device,
                    length_penalty=0.1
                )
        self.logits_processor = logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(16, eos_token_id=self.model.config.eos_token_id),
            ]
        )

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        
        label_ids = [f["labels"] for f in features]

        max_label_length = max(len(label_id) for label_id in label_ids)
        padded_label_ids = []
        for label_id in label_ids:
            padded_label_id = label_id + [self.tokenizer.pad_token_id] * (max_label_length - len(label_id))
            padded_label_ids.append(padded_label_id)
        label_ids = np.array(padded_label_ids)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        print(label_ids)

        decoded_label_ids = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        metas = [
            {
                "query": x["query"],
                "question": x["question"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]

        padded_predictions = []
        prev_prediction = None
        for prediction in predictions:
            if prev_prediction == None:
                prev_prediction = prediction
            else:
                if torch.allclose(prediction, prev_prediction):
                    print('same')
            prediction = prediction.cpu()
            padded_prediction = prediction.squeeze().tolist() + [self.tokenizer.pad_token_id] * (max_label_length - len(prediction))
            padded_predictions.append(padded_prediction)
        padded_predictions = np.array(padded_predictions)

        # predictions = predictions.argmax(axis=1)
        # predictions = softmax(predictions, axis=-1)
        # predictions = np.argmax(predictions, axis=-1)
        # print(predictions)
        # print(predictions.shape)

        decoded_predictions = self.tokenizer.batch_decode(padded_predictions, skip_special_tokens=True)
        
        assert len(metas) == len(decoded_predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w", encoding='utf-8') as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(decoded_predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=decoded_predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        # if self.target_with_db_id:
        #     Remove database id from all predictions
        #     predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        print(f'from seq/utils/trainer.py:- {eval_dataset}\n {eval_examples}')

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: EvalLoopOutput = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            ) # EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

            init_input_ids = torch.ones((8, 1), device=self.model.device, dtype=torch.long)
            init_input_ids = init_input_ids * self.model.config.decoder_start_token_id

            predictions = []
            self.model.eval()
            for instance in tqdm(eval_dataloader):
                model_kwargs = {
                    "encoder_outputs": self.model.get_encoder()(
                        instance['input_ids'].repeat_interleave(8, dim=0).to(self.model.device), return_dict=True
                    )
                }
                outputs = self.model.beam_search(
                    init_input_ids.to(self.model.device), 
                    self.beam_scorer, 
                    logits_processor=self.logits_processor, 
                    **model_kwargs
                )
                del model_kwargs
                predictions.append(outputs[0])
            
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                eval_dataset,
                predictions,
                "eval_{}".format(self.state.epoch),
            )
            output.metrics.update(self.compute_metrics(eval_preds))

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        # if self.target_with_db_id:
        #     # Remove database id from all predictions
        #     predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)