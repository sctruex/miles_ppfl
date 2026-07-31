# Miles: A System for Parameter Analysis in Practical, Private Federated Learning

This repository contains the source code and experimental configurations for:

S. Truex and M. Malan,
"Miles: A System for Parameter Analysis in Practical, Private Federated Learning."

## Scope

Miles is a customizable framework for systematic parameter analysis in practical privacy-preserving federated learning (PPFL).

## Paper Reproduction

All text files necessary to reproduce paper results are included within the experimental_params folder. Each configuration specifies the random seed and all experiment parameters.

## Datasets

The paper uses:

- MNIST
- Fashion-MNIST
- Adult
- PneumoniaMNIST

MNIST, Fashion-MNIST, and PneumoniaMNIST are obtained through their
respective dataset libraries. Adult preprocessing is provided in
`data_utils/csv_datasets/adult/`.

## Privacy

The experiments use RDP accounting. The clipping threshold and noise multiplier are adjusted as described in the paper.

## Clip Scheduler

The paper evaluates `StepClipDecay`:

C_r = C_0 * decay_rate^(floor(r / decay_rounds))

This scheduler is used as a representative case study for Miles and is not
claimed to be superior to private quantile-based adaptive clipping.

## Client Selection

Privacy-aware selection uses five buckets with:

q = (0.05, 0.10, 0.20, 0.25, 0.40)

Clients are selected uniformly within the sampled bucket.

## Reproducibility

Experiments use fixed random seeds. Independent parameter configurations and repeated runs may be executed in parallel.