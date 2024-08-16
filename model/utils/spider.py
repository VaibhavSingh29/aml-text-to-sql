import json
import anytree
from anytree import Node, RenderTree
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq.utils.dataset import DataTrainingArguments, normalize, serialize_schema
from seq2seq.utils.trainer import Seq2SeqTrainer, EvalPrediction
from model.utils.smbop import ra_preproc, ra_postproc
from model.utils.smbop import moz_sql_parser as msp 
import copy

POST_ORDER_LIST = []
import sys 
print(sys.getrecursionlimit())
sys.setrecursionlimit(3000)
class SmBopSQL2RA:
    def __init__(self) -> None:
        pass
    
    def generate_ast(self, sql_query):
        try: 
            sql_query = sql_query.lower()
            tree_dict = msp.parse(sql_query)
            return tree_dict['query']
        except msp.ParseException as e:
            print(f'Could not create AST for: {sql_query}')
            return None

    def ast_to_ra(self, query_ast):
        if not query_ast: return None
        tree_object = ra_preproc.ast_to_ra(query_ast)
        arit_list = anytree.search.findall(
                tree_object, filter_=lambda x: x.name in ["sub", "add"]
            )
        haslist_list = anytree.search.findall(
                    tree_object,
                    filter_=lambda x: hasattr(x, "val") and isinstance(x.val, list),
                )
        if arit_list or haslist_list:
            print(f'Could not create RA for:  {query_ast}')
            return None
        else:
            return tree_object
    
    def post_order(self, node):
        if  len(node.children) == 0:
            POST_ORDER_LIST.append('(')
            try:
                name = str(node.name) + '=' + str(node.val)
            except:
                name = str(node.name)
            POST_ORDER_LIST.append(name)
            POST_ORDER_LIST.append(')')
        elif len(node.children) == 1:
            POST_ORDER_LIST.append('(')
            self.post_order(node.children[0] if node.children[0] else node.children[1])
            try:
                name = str(node.name) + '=' + str(node.val)
            except:
                name = str(node.name)
            POST_ORDER_LIST.append(name)
            POST_ORDER_LIST.append(')')
        else:
            POST_ORDER_LIST.append('(')
            self.post_order(node.children[0])
            self.post_order(node.children[1])
            try:
                name = str(node.name) + '=' + str(node.val)
            except:
                name = str(node.name)
            POST_ORDER_LIST.append(name)
            POST_ORDER_LIST.append(')')
    
def spider_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


################ MODIFY THIS TO ADD RA TREE LINEAR SEQUENCE #############################

def spider_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    use_relational_algebra: bool,
    # target_with_db_id: bool,
) -> str:
    if use_relational_algebra:
        smbop = SmBopSQL2RA()
        ast = smbop.generate_ast(sql_query=query)
        if ast is None:
            return None
        tree_object = smbop.ast_to_ra(ast)
        if tree_object is None:
            return None
        global POST_ORDER_LIST
        POST_ORDER_LIST = []
        smbop.post_order(tree_object) 
        # print(POST_ORDER_LIST)
        return ''.join(POST_ORDER_LIST)
    else:
        _normalize = normalize if normalize_query else (lambda x: x)
        return _normalize(query)


def spider_add_serialized_schema(ex: dict, data_training_args: DataTrainingArguments) -> dict:
    serialized_schema = serialize_schema(
        question=ex["question"],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_randomized=data_training_args.schema_serialization_randomized,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
    )
    return {"serialized_schema": serialized_schema}


def spider_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""

    inputs = [
        spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
        for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
    ]

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_training_args.normalize_query,
            use_relational_algebra=data_training_args.use_relational_algebra,
            # target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    new_inputs = [input_item for input_item, target_item in zip(inputs, targets) if input_item is not None and target_item is not None]
    new_targets = [target_item for input_item, target_item in zip(inputs, targets) if input_item is not None and target_item is not None]

    # print(new_targets[0])

    model_inputs: dict = tokenizer(
        new_inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            new_targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"] # this is the features in _post_process_function
    print('TYPE OF MODEL_INPUTS', type(model_inputs["labels"]))
    return model_inputs
