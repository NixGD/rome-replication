import matplotlib.pyplot as plt
import numpy as np

from hook_handler import PatchCorruption, SaveAllActivations, PatchActivations
from gpt import GPT2
from utils import *


def run_baseline(model, input_ids, correct_id):
    with SaveAllActivations(model) as hh:
        logits = model(input_ids)
        correct_prob = get_correct_prob(logits, correct_id)
        return hh.get_activations(), correct_prob


def avg_evaluate(model, input_ids, correct_id, k=5, **kwargs):
    probs = []
    seeds = range(k)
    for seed in seeds:
        t.manual_seed(seed)
        corrupt_out = model(input_ids, **kwargs)
        probs.append(get_correct_prob(corrupt_out, correct_id))
    return sum(probs) / k


def patch_effectiveness_array(
    model, input_ids, patch_values, correct_id, k=3, corruption=None
):
    n_layers = len(model.blocks)
    n_tokens = input_ids.shape[1]
    assert patch_values.shape[:2] == (n_tokens, n_layers), patch_values.shape
    avg_prob = t.zeros((n_tokens, n_layers))

    for token_idx in range(n_tokens):
        for layer_idx in range(n_layers):
            patch_value = patch_values[token_idx, layer_idx, :]
            layer = model.blocks[layer_idx]
            embedding_layer = model.token_embedding
            with PatchActivations(layer, token_idx, patch_value):
                with PatchCorruption(embedding_layer, corruption):
                    prob = avg_evaluate(
                        model,
                        input_ids=input_ids,
                        correct_id=correct_id,
                        k=k,
                    )
            avg_prob[token_idx, layer_idx] = prob

    return avg_prob


def layer_token_plot(values, input_ids, tokenizer, cbar=True, **kwargs):
    if isinstance(values, t.Tensor):
        values = values.detach().to("cpu")
    plt.matshow(values)
    l = tokenizer.batch_decode([[id] for id in input_ids[0]])
    plt.yticks(ticks=range(input_ids.shape[1]), labels=[repr(t) for t in l])
    plt.xlabel("Patching activation at single layer")
    plt.gca().xaxis.set_label_position("top")
    if cbar:
        plt.colorbar()


def graph_patched_probs(model: GPT2, fact: Fact, k=3, noise_std=0.4, plot=True):
    tokenizer = model.tokenizer
    input_ids, subj_len, correct_id = fact_tensors(
        fact, tokenizer, device=get_device(model)
    )
    activations, p_baseline = run_baseline(model, input_ids, correct_id)

    corruption = Corruption(subj_len, noise_std)
    with PatchCorruption(model.token_embedding, corruption):
        p_corrupted = avg_evaluate(
            model,
            input_ids=input_ids,
            correct_id=correct_id,
            k=k,
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
            avg_probs,
            input_ids,
            tokenizer,
            vmin=p_corrupted,
            vmax=p_baseline,
            cmap="Blues",
        )

    return avg_probs
