from collections import namedtuple
import warnings
import torch as t
from tqdm import tqdm

import datasets

from gpt import GPT2
from hook_handler import HookHandler
from utils import *


def estimate_C(model: GPT2, layer):
    linear = model.blocks[layer].linear2

    dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1', split='train')
    dataset = dataset.map(lambda e: model.tokenizer(e['text'], truncation=True))
    dataset.set_format(type='torch', columns=['text', 'input_ids'])
    
    with HookHandler() as hh:
        hh.add_save_input_hook(linear)

        for input_ids in tqdm(dataset['input_ids'][:1000]):
            if input_ids.shape[0] > 0:
                input_ids = input_ids.unsqueeze(0).to(get_device(model))
                model(input_ids)
        
        input_tensor = t.cat(hh.inputs, dim=1).squeeze(0).T
    print(input_tensor.shape)
    C = t.cov(input_tensor)
    return C


def get_C(model, layer):
    cached = model.get_cached_covar_matrix(layer)
    if cached is None:
        C = estimate_C(model, layer)
        model.cache_covar_matrix(layer, C)
        return C
    else:
        print(f"Found cahced C for layer {layer}")
        return cached


def get_k_star_and_z0(model: GPT2, layer, fact):
    linear = model.blocks[layer].linear2

    with HookHandler() as hh:
        hh.add_save_input_output_hook(linear)
        input_ids = encode_for_model(model, fact.subject)
        model(input_ids)
        return hh.inputs[0][0,-1,:], hh.outputs[0][0,-1,:]


def get_z_star(model: GPT2, layer: int, fact: Fact, new_obj: str, z0: t.Tensor):
    tokenizer = model.tokenizer
    input_ids, subj_len, _ = fact_tensors(fact, tokenizer, get_device(model))
    new_obj_id = tokenizer.encode(new_obj, return_tensors="pt")
    assert len(new_obj_id) == 1
    new_obj_id = new_obj_id[0].to(z0.device)

    z = z0.clone().detach()
    z.requires_grad = True
    optim = t.optim.Adam([z], lr=.5)
    with tqdm.trange(50) as thandle:
        for step in thandle:
            optim.zero_grad()
            patch = Patch("mlp", subj_len, layer, z)
            out = model.forward_corrupt_and_patch(input_ids, patch=patch)
            new_obj_prob = out.logits.softmax(dim=-1)[0, new_obj_id]
            loss = -t.log(new_obj_prob)
            loss.backward()
            optim.step()

            thandle.set_postfix(loss=loss.item(), prob=new_obj_prob.item())
    return (z - model.blocks[layer].linear2.bias).detach() 




def calcuate_new_weights(W: t.Tensor, C: t.Tensor, k_star: t.Tensor, v_star: t.Tensor):
    """
    W: [hidden_size, 4*hidden_size]
    C: [4*hidden_size, 4*hidden_size]

    """
    hidden_size = W.shape[0]
    device = W.device

    # assert W.size == [hidden_size, 4 * hidden_size]

    u = t.linalg.solve(C, k_star)
    mat_1 = t.cat((W, v_star.unsqueeze(1)), dim=1)

    I = t.eye(4*hidden_size, device=device)
    first_rows = t.cat((I, k_star.unsqueeze(1)), dim=1)
    last_row = t.cat((-u.unsqueeze(0), t.zeros((1, 1), device=device)), dim=1)
    mat_2 = t.cat((first_rows, last_row), dim=0)
    W_hat = (mat_1 @ t.linalg.inv(mat_2))[:, :4 * hidden_size]
    return W_hat


def rome(model: GPT2, fact: Fact,  new_obj: str, layer: int, subject_pos: int = -1):
    linear = model.blocks[layer].linear2
    W = linear.weight
    print("Estimating C")
    C = get_C(model, layer)
    k_star, z0 = get_k_star_and_z0(model, layer, fact)
    print("Estimating v_star")
    v_star = get_z_star(model, layer, fact, new_obj, z0)

    W_hat = calcuate_new_weights(W, C, k_star, v_star)
    return W_hat