import torch
def mask_tokens(
    input_ids,
    vocab,
    mlm_probability=0.15,
    generator=None,
):
    labels = input_ids.clone()

    prob = torch.full(
        labels.shape,
        mlm_probability,
        device=input_ids.device,
    )

    special = (
        (input_ids == vocab["[CLS]"]) |
        (input_ids == vocab["[SEP]"]) |
        (input_ids == vocab["[PAD]"])
    )
    prob.masked_fill_(special, 0.0)

    masked = torch.bernoulli(prob, generator=generator).bool()
    labels[~masked] = -100

    replace = torch.bernoulli(
        torch.full(labels.shape, 0.9, device=input_ids.device),
        generator=generator,
    ).bool() & masked

    input_ids = input_ids.clone()
    input_ids[replace] = vocab["[MASK]"]

    return input_ids, labels
