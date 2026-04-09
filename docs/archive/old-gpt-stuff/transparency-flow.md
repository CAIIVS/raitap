# Transparency Resolution Flow

The transparency factory logic lives in `src/raitap/transparency/factory.py`. It is the
bridge between Hydra config, explainer objects, visualiser objects, and the final
artifacts written to disk or forwarded to tracking.

## Resolution flow

```text
transparency entry
  -> validate keys
  -> create explainer
  -> create visualisers
  -> validate backend and compatibility
  -> merge call kwargs
  -> resolve call data sources
  -> explainer.explain(...)
  -> ExplanationResult
```

## What the factory validates

The factory rejects malformed config early:

- unknown explainer keys outside `_target_`, `algorithm`, `constructor`, `call`, and
  `visualisers`
- unknown visualiser keys outside `_target_`, `constructor`, and `call`
- invalid `_target_` values that cannot be instantiated
- visualiser and algorithm combinations that are not declared compatible
- model objects that do not expose the expected backend adapter methods

## Why `constructor` and `call` are separate

RAITAP intentionally splits object construction from execution:

- `constructor` configures the explainer or visualiser instance itself
- `call` configures one execution of `compute_attributions()` or `visualise()`

This makes the Hydra surface clearer and lets the factory catch misplaced parameters
before a backend library fails deep inside its own code.

## Data-source resolution

Within `call`, values shaped like:

```yaml
background_data:
  source: imagenet_samples
  n_samples: 50
```

are treated as data-source references. The factory resolves these into tensors before
calling the explainer, so the backend receives runtime-ready data instead of raw config
mappings.

## Public API relationship

The public `raitap.transparency` package re-exports the main explainer and visualiser
classes. The factory is the implementation layer behind those public names, and the docs
therefore cover both:

- the public package API in the reference section
- the orchestration and validation behavior here in the explanation section
