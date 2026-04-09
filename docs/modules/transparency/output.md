# Output

```text
└── transparency/
    ├── captum_ig/                           # Output for the `captum_ig` explainer
    │   ├── attributions.pt                  # Raw attribution tensor
    │   ├── CaptumImageVisualiser_0.png      # Visualisation written by the first visualiser
    │   └── metadata.json                    # Explainer, algorithm, visualisers, and call kwargs
    └── captum_saliency/                     # One subdirectory per named explainer
        ├── attributions.pt                  # Raw attribution tensor
        ├── CaptumImageVisualiser_0.png      # Visualisation written by the first visualiser
        └── metadata.json                    # Explainer, algorithm, visualisers, and call kwargs
```

If an explainer uses more than one visualiser, RAITAP writes one PNG per
visualiser using the pattern `<VisualiserClassName>_<index>.png`.
