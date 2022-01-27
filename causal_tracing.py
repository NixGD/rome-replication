import matplotlib.pyplot as plt
import numpy as np

from hook_handler import HookHandler, SaveAllActivations
from gpt import GPT2
from utils import *


def run_baseline(model, input_ids, correct_id):
    with SaveAllActivations(model) as hh:
        logits = model(input_ids)
        correct_prob = get_correct_prob(logits, correct_id)
        return hh.get_activations, correct_prob


def avg_evaluate(model, input_ids, correct_id, k=5, **kwargs):
    probs = []
    seeds = range(k)
    for seed in seeds:
        t.manual_seed(seed)
        corrupt_out = model.forward_corrupt_and_patch(input_ids, **kwargs)
        probs.append(get_correct_prob(corrupt_out, correct_id))
    return sum(probs) / k


def patch_effectiveness_array(
    model, input_ids, patch_values, correct_id, k=3, corruption=None
):
    n_layers = len(model.blocks)
    n_tokens = input_ids.shape[1]
    assert patch_values.shape[:2] == (n_tokens, n_layers), patch_values.shape
    avg_prob = t.zeros((n_tokens, n_layers))

    for token in range(n_tokens):
        for layer in range(n_layers):
            patch_value = patch_values[token, layer, :]
            patch = Patch(type="act", token=token, layer=layer, value=patch_value)
            prob = avg_evaluate(
                model,
                input_ids=input_ids,
                correct_id=correct_id,
                k=k,
                patch=patch,
                corruption=corruption,
            )
            avg_prob[token, layer] = prob

    return avg_prob


def layer_token_plot(values, input_ids, tokenizer, cbar = True, **kwargs):
    if isinstance(values, t.Tensor):
        values = values.detach().to("cpu")
    plt.matshow(values)
    l = tokenizer.batch_decode([[id] for id in input_ids[0]])
    plt.yticks(ticks=range(input_ids.shape[1]), labels=[repr(t) for t in l])
    plt.xlabel("Patching activation at single layer")
    plt.gca().xaxis.set_label_position("top")
    if cbar:
        plt.colorbar()


def graph_patched_probs(
    model: GPT2, tokenizer, fact: Fact, k=3, noise_std=0.4, plot=True
):
    input_ids, subj_len, correct_id = fact_tensors(
        fact, tokenizer, device=get_device(model)
    )
    activations, p_baseline = run_baseline(model, input_ids, correct_id)

    corruption = Corruption(subj_len, noise_std)
    p_corrupted = avg_evaluate(
        model,
        input_ids=input_ids,
        correct_id=correct_id,
        k=k,
        corruption=corruption,
    )

    print(f"Input:")
    print_tokenized(input_ids[0], model.tokenizer)

    print(f"\nProb ability of the correct answer ({repr(fact.object)})")
    print(f"normal gpt: {p_baseline:.2%}")
    print(f"corrupted:  {p_corrupted:.2%}")

    avg_probs = patch_effectiveness_array(
        model, input_ids, activations, correct_id, k=3, corruption=corruption
    )

    if plot:
        layer_token_plot(
            avg_probs, input_ids, tokenizer, vmin=p_corrupted, vmax=p_baseline, cmap="Blues"
        )

    return avg_probs
