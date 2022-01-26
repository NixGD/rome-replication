import warnings
from collections import namedtuple
import torch as t

Corruption = namedtuple("Corruption", ["end_position", "noise_std"], defaults=[0.1])
Patch = namedtuple("Patch", ["type", "token", "layer", "value"])
Fact = namedtuple("Fact", ['subject', 'relation', 'object'])

def get_device(model):
    return next(model.parameters()).device

def encode_for_model(model, text):
    return model.tokenizer.encode(text, return_tensors="pt").to(get_device(model))

def fact_tensors(fact, tokenizer, device):
    if fact.relation[0] != " ":
        warnings.warn(f"The fact relation {fact.relation} does not start with a space")
    if fact.object[0] != " ":
        warnings.warn(f"The fact object {fact.object} does not start with a space")

    subject_ids = tokenizer.encode(fact.subject, return_tensors="pt").to(device)
    relation_ids = tokenizer.encode(fact.relation, return_tensors="pt").to(device)
    subj_len = subject_ids.shape[1]
    input_ids = t.cat((subject_ids, relation_ids), dim=1)

    correct_id = tokenizer.encode(fact.object)
    if len(correct_id) != 1:
        warnings.warn(
            f"The fact object {fact.object} is {len(correct_id)} tokens long, only using first token"
        )
    correct_id = correct_id[0]

    return input_ids, subj_len, correct_id

def most_likely(model, fact, k=5):
    input_ids, _, _ = fact_tensors(fact, model.tokenizer, device=get_device(model))

    model_out = model(input_ids)
    target_probs = t.softmax(model_out.logits.squeeze(0), dim=0)
    top_probs, top_ids = t.topk(target_probs, k=k)
    for i in range(k):
        token = model.tokenizer.decode(top_ids[i])
        print(f"{repr(token).ljust(15)}{top_probs[i]:.2%}")

def get_correct_prob(out, correct_id):
    return t.softmax(out.logits[0], dim=-1)[correct_id].item()

def print_tokenized(ids, tokenizer):
    l = tokenizer.batch_decode([[id] for id in ids])
    for t in l:
        print(repr(t), end=" ")
    print()
