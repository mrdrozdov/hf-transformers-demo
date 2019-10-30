import sys
import os
import json
import nltk

from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import torch
from transformers import *


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--batch', action='store_true')
parser.add_argument('--layer_choice', default=-1, type=int)
options = parser.parse_args()


print('Load model.')

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

# Models can return full list of hidden-states & attentions weights at each layer
model = model_class.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)
#input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
#all_hidden_states, all_attentions = model(input_ids)[-2:]
if options.cuda:
    model.cuda()


# Read data.

print('Read data.')

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

def flatten(lst):
    def helper(lst):
        if not isinstance(lst, (list, tuple)):
            yield lst
        else:
            for x in lst:
                for y in flatten(x):
                    yield y
    return [x for x in helper(lst)]

def subword_offsets(subword_units):
    res = []
    sofar = 0
    for x in subword_units:
        size = len(x)
        res.append((sofar, size))
        sofar += size
    return res

with open('22.auto.clean') as f:
    for line in f:
        nltk_tree = nltk.Tree.fromstring(line)
        tokens = nltk_tree.leaves()
        rich_labeled_spans = get_spans(nltk_tree)
        labeled_spans = [(pos, size, labels[0]) for pos, size, labels in rich_labeled_spans]
        input_ids = tokenizer.encode(' '.join(tokens), add_special_tokens=True)
        subword_units = [tokenizer.encode(x) for x in tokens]
        subword_ids = flatten(subword_units)
        offsets = subword_offsets(subword_units)
        assert len(offsets) == len(tokens)
        assert tuple(input_ids[1:-1]) == tuple(subword_ids)
        dataset.append(dict(labeled_spans=labeled_spans, tokens=tokens, input_ids=input_ids, offsets=offsets))

# Encode.

print('Encode.')

## Sort by token length.
dataset = sorted(dataset, key=lambda x: len(x['input_ids']))

max_batch_size = 128

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

layer_choice = options.layer_choice
example_ids = []
vectors = []
labels = []
for i, batch in tqdm(enumerate(batch_iterator(dataset)), disable=False):
    input_ids = torch.tensor([x['input_ids'] for x in batch])
    if options.cuda:
        input_ids = input_ids.cuda()
    batch_size, length = input_ids.shape

    # Encode.
    all_hidden_states, all_attentions = model(input_ids)[-2:]
    size = all_hidden_states[0].shape[-1]
    h = all_hidden_states[layer_choice].view(batch_size, length, size)
    if options.random:
        h = h.normal_()
    h = h.cpu().data.numpy()
    #h = all_hidden_states[layer_choice].view(batch_size*length, -1)
    #vectors.append(h.cpu().data.numpy())

    # Save example ids.
    #batch_example_ids = torch.LongTensor(batch_size, length)
    #for j in range(batch_size):
    #    batch_example_ids[j] = i * batch_size + j
    #example_ids.append(batch_example_ids.view(-1))

    # All phrases.
    batch_example_ids = []
    batch_labels = []
    batch_vec = []
    for j, subbatch in enumerate(batch):
        offsets = subbatch['offsets']
        labeled_spans = subbatch['labeled_spans']
        for pos, size, label in labeled_spans:
            if size < 2:
                continue
            special_offset = 1
            start = special_offset + offsets[pos][0]
            end = special_offset + offsets[pos + size - 1][0] + offsets[pos + size - 1][1]
            boundary_start = h[j, start - 1]
            boundary_end = h[j, end]
            internal = np.mean(h[j, start:end], axis=0)
            vec = np.concatenate([boundary_start, boundary_end, internal], axis=-1).reshape(1, 768 * 3)
            batch_example_ids.append(i * batch_size + j)
            batch_labels.append(label)
            batch_vec.append(vec)

    example_ids.append(batch_example_ids)
    vectors.append(batch_vec)
    labels.append(batch_labels)

example_ids = flatten(example_ids)
vectors = flatten(vectors)
labels = flatten(labels)

print('# of phrases', len(example_ids))
assert len(example_ids) == len(vectors)
assert len(example_ids) == len(labels)

# Cluster.

print('Cluster.')

n_clusters = 25
seed = 11

X = np.concatenate(vectors, axis=0)
algo = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1600)
#algo = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=8, verbose=1)
algo.fit(X)

cluster_ids = algo.labels_

import collections

summary = collections.OrderedDict()

for idx in range(n_clusters):
    summary[idx] = collections.Counter()

for idx, y in zip(cluster_ids, labels):
    summary[idx][y] += 1

total_correct = 0
total_n = 0
for idx, counter in summary.items():
    if len(counter) == 0:
        print('cluster = {}, skipped'.format(idx))
    n = sum(counter.values())
    most_common = counter.most_common(3)
    correct = most_common[0][1]
    correct_label = most_common[0][0]

    print('cluster = {} {} {}/{}, {}'.format(
        idx, correct_label, correct, n, most_common))

    total_correct += correct
    total_n += n

print('Accuracy = {:.3f} {}/{}'.format(total_correct/total_n, total_correct, total_n))

