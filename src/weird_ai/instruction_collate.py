"""
Custom collate function for Weird AI instruction fine-tuning.
"""

import torch


def custom_collate_fn(
    batch,
    pad_token_id=0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Pad variable-length token ID lists, create shifted targets, and mask extra padding.
    """

    # TODO:
    # Find the maximum length in the batch.
    # Add 1 because we append a padding/end token before creating targets.
    batch_max_length = max(len(item) for item in batch) + 1

    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()

        # TODO: append one pad token to new_item.
        new_item.append(pad_token_id)

        # TODO: pad new_item to batch_max_length.
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # TODO: inputs are padded[:-1], targets are padded[1:].
        inputs = padded[:-1]
        targets = padded[1:]

        # TODO: replace all but the first pad_token_id in targets with ignore_index.
        pad_positions = [i for i, token in enumerate(targets) if token == pad_token_id]
        for i in pad_positions[1:]:
            targets[i] = ignore_index

        # TODO: if allowed_max_length is not None, truncate inputs and targets.
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # TODO: stack input and target tensors, then move them to device.
    inputs_tensor = torch.tensor(inputs_lst, dtype=torch.long).to(device)
    targets_tensor =  torch.tensor(targets_lst, dtype=torch.long).to(device)

    return inputs_tensor, targets_tensor
