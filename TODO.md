# Immediate TODOs 2025-09-26

- [ ] Fix the mixed precision training
- [ ] Do _NOT_ compute validation loss (mean / std / CV) per batch, but over samples
- [ ] Plot gradient norms to detect exploding gradients
- [ ] Investigate whether gradient clipping helps

- [ ] LR scheduling: BERT fine‑tuning commonly uses linear warmup + decay. Add
      a scheduler (e.g., get_linear_schedule_with_warmup) for smoother training.
- [ ] Consider no‑decay params (bias, LayerNorm.weight) to avoid regularizing them.

- [ ] Early stopping
- [ ] Regularization:
    - [ ] Weight decay via L2
    - [ ] Increase dropout rate
- [ ] Consider cross-validation
- [ ] Fix seeds for reproducibility
- [ ] Log everything 
    - Check if wandb can help with this


## Notes

### Computing loss metrics

When validating a model, the validation data is usually split into batches,
and the loss is calculated for each batch (≠ per sample).

Additionally, these losses are averaged over the batch losses to get the overall
validation loss for the epoch. Standard deviation (std) of the batch losses or
CV may also be calculated. This is basically measuring how much the loss varies
_between_ batches

**Now, the issue with this approach**: This can inflate the loss metrics, because
each batch might have very different difficulty levels or sizes. So the loss on
one batch might be much higher or lower than on another. This variation between
batches shows up as a large std.

So, I really want the standard deviation of the loss across all individual samples
in the entire validation set, not just batch-level variation.

In PyTorch, I will have to pass `reduction='none'` to the loss constructor to not
get the mean loss per batch (default).

> Do I only have to do this during validation, or also for training?

No, only for validation

Same applies for testing.

> How do I update my weights then?

See answer of the question above. During training, not necessary...

But I could still do:

```
losses = criterion(outputs, labels)
loss = losses.mean()
# then backprop
```

> How do I keep track of this?

1. Calculate loss per sample with `reduction='none'`
2. Accumulate all losses of all batches in a list for that epoch
3. Compute and store the mean, std, and CV. Free the list for the next epoch.

> More detailed loss curves...

Instead of saving _every_ single sample loss of the validation set,
I could also compute the mean loss per batch (or per N samples) and store
that in addition to the per-batch mean loss.