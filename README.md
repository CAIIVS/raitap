[![PyPI - Version](https://img.shields.io/pypi/v/raitap)](https://pypi.org/project/raitap/) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/CAIIVS/raitap)

<img width="400" alt="logo" src="https://github.com/user-attachments/assets/9de86f66-f85d-47b4-9bce-7ca10bb14406" />

RAITAP is a Python library to assess the responsibility level of AI models. It is designed to be easily integrated into existing MLOps workflows.

It is a wrapper around existing XAI frameworks, which provides a consistent API, allowing you to easily switch your configuration, combine frameworks, and obtain consolidated outputs.

RAITAP currently assesses the following 2 responsible AI dimensions:

- Transparency
- Robustness

as defined in [Towards the certification of AI-based systems](https://doi.org/10.1109/SDS60720.2024.00020) and [MLOps as enabler of trustworthy AI](https://doi.org/10.1109/SDS60720.2024.00013)

For more information

- as a consumer/user: [consumer documentation](https://caiivs.github.io/raitap/)
- as a contributor: [contributor documentation](https://caiivs.github.io/raitap/contributor/index.html)

## Showcase

<table>
  <tr>
    <td width="50%"><img src="docs/_static/showcase-terminal.png" alt="raitap CLI run in a terminal" /></td>
    <td width="50%"><img src="docs/_static/showcase-report.jpeg" alt="Generated raitap report" /></td>
  </tr>
  <tr>
    <td align="center"><sub>CLI run</sub></td>
    <td align="center"><sub>Generated report</sub></td>
  </tr>
</table>
