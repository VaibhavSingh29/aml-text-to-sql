import sys
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict, fields

from model.utils.dataset_loader import load_dataset
from model.utils.dataset import DataTrainingArguments, DataArguments
from model.utils.args import ModelArguments

from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint, set_seed

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from tokenizers import AddedToken

from transformers.hf_argparser import HfArgumentParser
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from tokenizers import AddedToken
from transformers.models.auto import AutoConfig, AutoTokenizer # WHAT IS AUTOCONFIG USED FOR?
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from model.seq2seq.trainer import T5SpiderTrainer
from transformers.models.auto import AutoModelForSeq2SeqLM

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )

    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, data_training_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, data_training_args, training_args = parser.parse_args_into_dataclasses()

    if 'checkpoint-???' in model_args.model_name_or_path:
        model_args.model_name_or_path = get_last_checkpoint(
            os.path.dirname(model_args.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {model_args.model_name_or_path}")

    combined_args_dict = {
        # **asdict(picard_args),
        **asdict(model_args),
        **asdict(data_args),
        **asdict(data_training_args),
        **training_args.to_sanitized_dict(),
    }

    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.local_rank <= 0:
        with open(f"{training_args.output_dir}/combined_args.json", "w") as f:
            json.dump(combined_args_dict, f, indent=4)

    # Initialize random number generators
    set_seed(training_args.seed)

    # Initialize config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
        gradient_checkpointing=training_args.gradient_checkpointing,
        use_cache=not training_args.gradient_checkpointing,
    )


    # Initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    # Load dataset
    metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    try:
        for instance in dataset_splits.train_split.dataset:
            print(instance)
            print(tokenizer.batch_decode(instance['input_ids']))
            # Split the decoded output into a list of tokens
            decoded_tokens = tokenizer.convert_ids_to_tokens(instance['input_ids'])
            # Join the tokens into a space-separated string
            decoded_text = ''.join(decoded_tokens)
            print(decoded_text)
            print(instance['labels'])
            print(''.join(tokenizer.convert_ids_to_tokens(instance['labels'])))
            break
    except:
        pass

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if isinstance(model, T5ForConditionalGeneration):
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f'LEN OF TOKEniZER: {len(tokenizer)}')

    # Initialize Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "metric": metric,
        "train_dataset": dataset_splits.train_split.dataset if training_args.do_train else None,
        "eval_dataset": dataset_splits.eval_split.dataset if training_args.do_eval else None,
        "eval_examples": dataset_splits.eval_split.examples if training_args.do_eval else None,
        "tokenizer": tokenizer,
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=(-100 if data_training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
            pad_to_multiple_of=8 if training_args.fp16 else None,
        ),
        "ignore_pad_token_for_loss": data_training_args.ignore_pad_token_for_loss,
    }

    trainer = T5SpiderTrainer(**trainer_kwargs)

    if training_args.do_train:
        logger.info('*** Train ***')
        checkpoint=None

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            logger.info(f'Resuming from {last_checkpoint}')
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_training_args.max_train_samples
            if data_training_args.max_train_samples is not None
            else len(dataset_splits.train_split.dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(dataset_splits.train_split.dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_training_args.val_max_target_length,
            max_time=data_training_args.val_max_time,
            num_beams=data_training_args.num_beams,
            metric_key_prefix="eval",
        )
        max_val_samples = (
            data_training_args.max_val_samples
            if data_training_args.max_val_samples is not None
            else len(dataset_splits.eval_split.dataset)
        )
        metrics["eval_samples"] = min(max_val_samples, len(dataset_splits.eval_split.dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()



