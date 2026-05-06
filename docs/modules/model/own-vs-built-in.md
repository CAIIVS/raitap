# Using your own model or built-in models

## Own model

RAITAP allows you to use your own model in any of the following supported formats:

- `.pt` / `.pth` containing a `state_dict` — **recommended**. Combine with
  `model.arch` and `model.num_classes` to instantiate a torchvision
  architecture and load weights into it. Portable across environments and
  torchvision versions.
- `.pt` / `.pth` containing a TorchScript archive (saved via
  `torch.jit.save(scripted, path)`). Self-contained — no `arch` /
  `num_classes` needed.
- `.pt` / `.pth` containing a full pickled `torch.nn.Module` (saved via
  `torch.save(model, path)`). **Deprecated** — emits a
  `DeprecationWarning`. The pickle embeds fully-qualified class paths so it
  breaks when classes are renamed or when torchvision is bumped. Migrate
  with one line:
  ```python
  m = torch.load("model.pth", weights_only=False)
  torch.save(m.state_dict(), "weights.pth")
  ```
- `.onnx`

Set the `source` option to the path of your model file (see
{doc}`configuration`). For state-dict loading also set `arch` and
`num_classes`:

```yaml
model:
  source: "weights.pth"
  arch: "resnet18"
  num_classes: 2
```

Not all modules and underlying libraries support all formats. See each
module's library compatibility page for more details.

## Built-in models

Alternatively, you can use any model provided by the `torchvision.models` library. They will be initialised with `weights="DEFAULT"`. You can [find the list here](https://docs.pytorch.org/vision/stable/models.html).

You simply need to set the `source` option to the name of the model (see {doc}`configuration`).
