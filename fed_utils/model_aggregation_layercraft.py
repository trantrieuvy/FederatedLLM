"""
Federated averaging (FedAvg) for layercraft adapters.

Drop-in replacement for model_aggregation.py — supports:
  1. Homogeneous:  all clients same rank → weighted average
  2. Heter + zero_padding: different ranks → pad to max, then average
  3. Heter + stacking: different ranks → concatenate along rank dim

Uses layercraft.load_adapter_state_dict instead of PEFT's set_peft_model_state_dict.
"""

import torch
import os
from torch.nn.functional import normalize
import layercraft


def _zero_pad_tensor(tensor, client_rank, max_rank):
    """
    Zero-pad every dimension of `tensor` that matches `client_rank` up to `max_rank`.

    Handles all layercraft weight shapes:
      - A  (r, in)       → pad dim 0
      - B  (out, r)      → pad dim 1
      - T  (r, r)        → pad both dims
      - diag (r,)        → pad dim 0
    """
    if client_rank == max_rank:
        return tensor
    for dim in range(tensor.ndim):
        if tensor.shape[dim] == client_rank:
            pad_amount = max_rank - client_rank
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_amount
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=dim)
    return tensor


def _stack_tensor(existing, new_tensor, client_rank):
    """
    Concatenate `new_tensor` onto `existing` along every dimension that
    matches `client_rank`.

    For rank-dim stacking:
      - A  (r, in)  → cat along dim 0   → (r1+r2, in)
      - B  (out, r) → cat along dim 1   → (out, r1+r2)
      - T  (r, r)   → block-diagonal extension
      - diag (r,)   → cat along dim 0   → (r1+r2,)
    """
    for dim in range(new_tensor.ndim):
        if new_tensor.shape[dim] == client_rank:
            # For square matrices (r, r), we need to pad the OTHER dims
            # of new_tensor to match existing before cat.
            # E.g., existing is (r1, r1), new is (r2, r2).
            # To cat along dim 0: pad new to (r2, r1) first → then cat → (r1+r2, r1).
            # Then cat along dim 1 in the next iteration.
            if existing.shape[dim] != new_tensor.shape[dim]:
                # new_tensor's other dims might not match existing yet;
                # pad the non-cat dims of new_tensor to match existing
                for other_dim in range(new_tensor.ndim):
                    if other_dim != dim and new_tensor.shape[other_dim] < existing.shape[other_dim]:
                        pad_amount = existing.shape[other_dim] - new_tensor.shape[other_dim]
                        pad_shape = list(new_tensor.shape)
                        pad_shape[other_dim] = pad_amount
                        padding = torch.zeros(pad_shape, dtype=new_tensor.dtype, device=new_tensor.device)
                        new_tensor = torch.cat([new_tensor, padding], dim=other_dim)

            existing = torch.cat([existing, new_tensor], dim=dim)
    return existing


def FedAvg(
    model,
    selected_clients_set,
    output_dir,
    local_dataset_len_dict,
    epoch,
    stacking,
    lora_r,
    heter,
    local_ranks,
    zero_padding,
    nonlinear=False,
):
    """
    Aggregate client adapter weights via FedAvg.

    Args:
        model: The global model (with layercraft adapters already injected).
        selected_clients_set: Set of client IDs that participated this round.
        output_dir: Base output directory.
        local_dataset_len_dict: {client_id: dataset_size} for weighting.
        epoch: Current communication round.
        stacking: If True, concatenate adapters along rank dim.
        lora_r: Base LoRA rank (used for homogeneous mode).
        heter: If True, clients have heterogeneous ranks.
        local_ranks: List of per-client ranks.
        zero_padding: If True, pad smaller ranks to max before averaging.

    Returns:
        model with aggregated adapter weights loaded.
    """
    # Compute dataset-size-based weights for averaging
    weights_array = normalize(
        torch.tensor(
            [local_dataset_len_dict[client_id] for client_id in selected_clients_set],
            dtype=torch.float32,
        ),
        p=1,
        dim=0,
    )
    print("Weights:", weights_array)

    weighted_single_weights = None

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(
            output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin"
        )
        single_weights = torch.load(single_output_dir, map_location="cpu")
        weight = weights_array[k]

        if stacking:
            # ---- Stacking: concatenate along rank dimension ----
            # Linear adapters (default): weight A (rank in dim 0), leave B unweighted.
            #   B_stacked @ (pk * A_stacked) @ x = pk * B @ A @ x  ✓
            # Nonlinear adapters (nonlinear=True): weight B (rank in dim 1), leave A unweighted.
            #   (pk * B_stacked) @ σ(A_stacked @ x) = pk * B @ σ(A @ x)  ✓
            #   Weighting A would bury pk inside σ, distorting the intended scaling.
            client_rank = local_ranks[client_id] if heter else lora_r
            if k == 0:
                weighted_single_weights = {}
                for key in single_weights:
                    t = single_weights[key]
                    is_A = t.ndim >= 1 and t.shape[0] == client_rank
                    is_B = (not is_A) and t.ndim >= 2 and t.shape[1] == client_rank
                    if nonlinear:
                        # Nonlinear: weight B, leave A unweighted
                        if is_B:
                            weighted_single_weights[key] = t * weight
                        else:
                            weighted_single_weights[key] = t.clone()
                    else:
                        # Linear: weight A, leave B unweighted
                        if is_A:
                            weighted_single_weights[key] = t * weight
                        else:
                            weighted_single_weights[key] = t.clone()
            else:
                for key in single_weights:
                    t = single_weights[key]
                    is_A = t.ndim >= 1 and t.shape[0] == client_rank
                    is_B = (not is_A) and t.ndim >= 2 and t.shape[1] == client_rank
                    if is_A:
                        if nonlinear:
                            # Nonlinear: A unweighted, stack along dim 0
                            weighted_single_weights[key] = torch.cat(
                                [weighted_single_weights[key], t], dim=0
                            )
                        else:
                            # Linear: A weighted, stack along dim 0
                            weighted_single_weights[key] = torch.cat(
                                [weighted_single_weights[key], t * weight], dim=0
                            )
                    elif is_B:
                        if nonlinear:
                            # Nonlinear: B weighted, stack along dim 1
                            weighted_single_weights[key] = torch.cat(
                                [weighted_single_weights[key], t * weight], dim=1
                            )
                        else:
                            # Linear: B unweighted, stack along dim 1
                            weighted_single_weights[key] = torch.cat(
                                [weighted_single_weights[key], t], dim=1
                            )
                    else:
                        # Non-adapter param (e.g. bias): weighted average
                        weighted_single_weights[key] += t * weight

        elif zero_padding and heter:
            # ---- Zero-padding: pad to max rank, then weighted average ----
            max_rank = max(local_ranks)
            client_rank = local_ranks[client_id]

            if k == 0:
                weighted_single_weights = {}
                for key in single_weights:
                    padded = _zero_pad_tensor(single_weights[key], client_rank, max_rank)
                    weighted_single_weights[key] = padded * weight
            else:
                for key in single_weights:
                    padded = _zero_pad_tensor(single_weights[key], client_rank, max_rank)
                    weighted_single_weights[key] += padded * weight

        else:
            # ---- Homogeneous: simple weighted average ----
            if k == 0:
                weighted_single_weights = {
                    key: single_weights[key] * weight for key in single_weights
                }
            else:
                for key in single_weights:
                    weighted_single_weights[key] += single_weights[key] * weight

    # Apply aggregated weights to the global model
    if stacking:
        # For stacking, save the stacked weights — the main script will handle
        # creating a new model with the stacked rank
        torch.save(
            weighted_single_weights,
            os.path.join(output_dir, str(epoch), "adapter_model.bin"),
        )
    else:
        layercraft.load_adapter_state_dict(model, weighted_single_weights)

    return model
