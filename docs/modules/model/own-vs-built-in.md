# Using your own model or built-in models

## Own model

RAITAP allows you to use your own model, in any of the following supported formats:

- `.pth` (if it contains a `torch.nn.Module` and not just a state-dict)
- `.pt` (if it contains a `torch.nn.Module` and not just a state-dict)
- `.onnx`

You simply need to set the `source` option to the path to your model file (see {doc}`configuration`).

However, not all modules and underlying libraries support all formats. See each module's library compatibility page for more details.

## Built-in models

Alternatively, you can use any model provided by the `torchvision.models` library. They will be initialised with `weights="DEFAULT"`. You can [find the list here](https://docs.pytorch.org/vision/stable/models.html).

You simply need to set the `source` option to the name of the model (see {doc}`configuration`).
