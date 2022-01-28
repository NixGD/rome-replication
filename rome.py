from collections import namedtuple
import warnings
import torch as t
import tqdm

import datasets

from gpt import GPT2
from hook_handler import HookHandler
from utils import *


def estimate_C(model: GPT2, layer):
    linear = model.blocks[layer].linear2

    dataset = datasets.load_dataset("wikitext", "wikitext-2-v1", split="train")
    dataset = dataset.map(lambda e: model.tokenizer(e["text"], truncation=True))
    dataset.set_format(type="torch", columns=["text", "input_ids"])

    with HookHandler() as hh:
        hh.add_save_input_hook(linear)

        for input_ids in tqdm.tqdm(dataset["input_ids"][:2000]):
            if input_ids.shape[0] > 0:
                input_ids = input_ids.unsqueeze(0).to(get_device(model))
                model(input_ids)

        input_tensor = t.cat(hh.inputs, dim=1).to(get_device(model)).squeeze(0).T
    print(input_tensor.shape)
    C = t.cov(input_tensor)
    return C


def get_C(model, layer, verbose=True):
    cached = model.get_cached_covar_matrix(layer)
    if cached is None:
        C = estimate_C(model, layer)
        model.cache_covar_matrix(layer, C)
        return C
    else:
        if verbose:
            print(f"Found cahced C for layer {layer}")
        return cached


def get_k_star_and_z0(model: GPT2, layer, fact, subj_pos):
    linear = model.blocks[layer].linear2

    with HookHandler() as hh:
        hh.add_save_input_output_hook(linear)
        input_ids = encode_for_model(model, fact.subject)
        model(input_ids)
        return hh.inputs[0][0, subj_pos, :], hh.outputs[0][0, subj_pos, :]


def get_v_star(
    model: GPT2,
    layer: int,
    fact: Fact,
    new_obj: str,
    z0: t.Tensor,
    subj_pos=-1,
    reg_coeff=0.02,
):
    """
    Notation point:
    Following the paper -- we will call this variable z while minimizing the loss,
    but once we find the optimial value we will denote it v_star.
    """
    tokenizer = model.tokenizer
    input_ids, subj_len, _ = fact_tensors(fact, tokenizer, get_device(model))
    new_obj_id = tokenizer.encode(new_obj, return_tensors="pt")
    assert len(new_obj_id) == 1
    new_obj_id = new_obj_id[0].to(z0.device)

    if subj_pos < 0:
        subj_pos = subj_len + subj_pos + 1

    z = z0.clone().detach()
    z.requires_grad = True
    optim = t.optim.Adam([z], lr=0.05)
    with tqdm.trange(200) as thandle:
        for step in thandle:
            optim.zero_grad()
            patch = Patch("mlp", subj_pos, layer, z)
            out = model.forward_corrupt_and_patch(input_ids, patch=patch)
            new_obj_prob = out.logits.softmax(dim=-1)[0, new_obj_id]
            prob_loss = -t.log(new_obj_prob)
            reg_loss = t.linalg.vector_norm(z0 - z) * reg_coeff
            loss = prob_loss + reg_loss
            loss.backward()
            optim.step()

            thandle.set_postfix(
                loss=loss.item(), reg_loss=reg_loss.item(), prob=new_obj_prob.item()
            )
    return (z - model.blocks[layer].linear2.bias).detach()


def calcuate_new_weights(W: t.Tensor, C: t.Tensor, k_star: t.Tensor, v_star: t.Tensor):
    """
    W: [hidden_size, 4*hidden_size]
    C: [4*hidden_size, 4*hidden_size]

    """
    hidden_size = W.shape[0]
    device = W.device

    # assert W.size == [hidden_size, 4 * hidden_size]

    u = t.linalg.solve(C, k_star.to(device))
    mat_1 = t.cat((W, v_star.unsqueeze(1)), dim=1)

    I = t.eye(4 * hidden_size, device=device)
    first_rows = t.cat((I, k_star.unsqueeze(1).to(device)), dim=1)
    last_row = t.cat((-u.unsqueeze(0), t.zeros((1, 1), device=device)), dim=1)
    mat_2 = t.cat((first_rows, last_row), dim=0)
    W_hat = (mat_1 @ t.linalg.inv(mat_2))[:, : 4 * hidden_size]
    return W_hat


def rome(model: GPT2, fact: Fact, new_obj: str, layer: int, subj_pos: int = -1, reg_coeff=0.02, v_star=None, verbose=True):
    linear = model.blocks[layer].linear2
    W = linear.weight
    if verbose:
        print("Estimating C")
    C = get_C(model, layer, verbose=verbose)
    k_star, z0 = get_k_star_and_z0(model, layer, fact, subj_pos)
    if v_star is None:
        if verbose:
            print("Estimating v_star")
        v_star = get_v_star(model, layer, fact, new_obj, z0, subj_pos, reg_coeff=reg_coeff)
    elif verbose:
        print("Using given v_star")
    W_hat = calcuate_new_weights(W, C, k_star, v_star)
    return W_hat


class ModifyWeights(HookHandler):
    def __init__(self, model, layer, new_weights):
        super().__init__()
        self.model = model
        self.layer = layer
        self.new_weights = new_weights

    def __enter__(self):
        linear = self.model.blocks[self.layer].linear2
        self.add_hook(linear, get_edit_hook(self.new_weights))

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset()
        # print("All hooks removed!")
