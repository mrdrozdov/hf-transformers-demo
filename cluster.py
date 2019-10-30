import sys
import os
import json
import nltk

import torch
from transformers import *

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]
#MODELS = [(BertModel,       BertTokenizer,       'bert-base-cased')]
MODELS = [(BertModel,       BertTokenizer,       'spanbert_hf_base')]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
                      BertForQuestionAnswering]
BERT_MODEL_CLASSES = [BertModel]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
#pretrained_weights = 'bert-base-cased'
pretrained_weights = 'spanbert_hf_base'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model_class = BERT_MODEL_CLASSES[0]

# Load pretrained model/tokenizer
model = model_class.from_pretrained(pretrained_weights)

# Models can return full list of hidden-states & attentions weights at each layer
model = model_class.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)
#input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
#all_hidden_states, all_attentions = model(input_ids)[-2:]


def get_spans(tr):
    spans = {}

    def helper(tr, pos):
        if not isinstance(tr, str) and len(tr) == 1 and isinstance(tr[0], str):
            size = 1
            spans.setdefault((pos, size), []).append(tr.label())
            return size
        if not isinstance(tr, str) and len(tr) == 1:
            size = helper(tr[0], pos)
            spans.setdefault((pos, size), []).append(tr.label())
            return size
        size = 0
        for x in tr:
            xsize = helper(x, pos+size)
            size += xsize
        spans.setdefault((pos, size), []).append(tr.label())
        return size
    helper(tr, 0)
    return [(pos, size, labels) for (pos, size), labels in spans.items()]

dataset = []

with open('22.auto.clean') as f:
    for line in f:
        nltk_tree = nltk.Tree.fromstring(line)
        tokens = nltk_tree.leaves()
        rich_labeled_spans = get_spans(nltk_tree)
        labeled_spans = [(pos, size, labels[0]) for pos, size, labels in rich_labeled_spans]
        input_ids = tokenizer.encode(' '.join(tokens))
        dataset.append(dict(labeled_spans=labeled_spans, tokens=tokens, input_ids=input_ids))

# Sort by token length.
dataset = sorted(dataset, key=lambda x: len(x['input_ids']))

max_batch_size = 32

def batch_iterator(dataset):
    batch = []
    length = None
    for x in dataset:
        xlength = len(x['input_ids'])
        if length is None:
            length = xlength
        assert length <= xlength
        if length == xlength:
            batch.append(x)
            if len(batch) == max_batch_size:
                yield batch
                batch = []
            continue
        if len(batch) > 0:
            yield batch
            batch = []
        length = xlength
    if len(batch) > 0:
        yield batch

for batch in batch_iterator(dataset):
    input_ids = torch.tensor([x['input_ids'] for x in batch])
    print(input_ids.shape)
import ipdb; ipdb.set_trace()
pass

