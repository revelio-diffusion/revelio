### Supported Models and Datasets

This code currently supports models and datasets hosted on Hugging Face.

#### Getting Started
Refer to the [`run.sh`](./run.sh) script for an example run.

#### Additional Details
- **Custom Prompts**: To use specific prompts for feature extraction, refer to [`helpers/prompt_dict.py`](./helpers/prompt_dict.py).
- **Adding Models**: Add new models to the `model_base_dict` dictionary in [`constants.py`](./constants.py).
