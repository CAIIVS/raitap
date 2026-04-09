# Using your own model or built-in models

## Own model

RAITAP allows you to use your own model, in any of the following supported formats:

- `.pth` (if it contains a `torch.nn.Module` and not just a state-dict)
- `.pt` (if it contains a `torch.nn.Module` and not just a state-dict)
- `.onnx`

You simply need to set the `source` option to the path to your model file (see [Configuration](configuration.md)).

## Built-in models

Alternatively, you can use any model provided by the `torchvision.models` library. They will be initialised with `weights="DEFAULT"`.

You simply need to set the `source` option to the name of the model (see [Configuration](configuration.md)).
