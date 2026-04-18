# Output

```text
└── transparency/
    ├── captum_ig/                           # Output for the `captum_ig` explainer
    │   ├── attributions.pt                  # Raw attribution tensor
    │   ├── CaptumImageVisualiser_0.png      # Visualisation written by the first visualiser
    │   └── metadata.json                    # Explainer metadata plus separate call_kwargs
    └── captum_saliency/                     # One subdirectory per named explainer
        ├── attributions.pt                  # Raw attribution tensor
        ├── CaptumImageVisualiser_0.png      # Visualisation written by the first visualiser
        └── metadata.json                    # Explainer metadata plus separate call_kwargs
```

If an explainer uses more than one visualiser, RAITAP writes one PNG per
visualiser using the pattern `<VisualiserClassName>_<index>.png`.

`metadata.json` now stores two separate buckets:

- `kwargs`: RAITAP-owned metadata used for downstream visualisation, such as
  `sample_names` and `show_sample_names`
- `call_kwargs`: library invocation parameters such as `target`, `baselines`,
  `background_data`, or `nsamples`

`call_kwargs` is a best-effort JSON summary of the library invocation. Scalar
values are stored directly, while tensor-like values are summarized rather than
embedded verbatim so `metadata.json` stays lightweight and readable.
