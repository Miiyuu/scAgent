# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['scgpt',
 'scgpt.model',
 'scgpt.scbank',
 'scgpt.tasks',
 'scgpt.tokenizer',
 'scgpt.utils']

package_data = \
{'': ['*']}

install_requires = \
['cell-gears<0.0.3',
 'datasets>=2.3.0,<3.0.0',
 'leidenalg>=0.8.10',
 'numba>=0.55.1',
 'orbax<0.1.8',
 'pandas>=1.3.5',
 'scanpy>=1.9.1,<2.0.0',
 'scib>=1.0.3,<2.0.0',
 'scikit-misc>=0.1.4',
 'scvi-tools>=0.16.0,<1.0',
 'torch>=1.13.0,<2.2',
 'torchtext',
 'typing-extensions>=4.2.0,<5.0.0',
 'umap-learn>=0.5.3']

setup_kwargs = {
    'name': 'scgpt',
    'version': '0.2.1',
    'description': 'Large-scale generative pretrain of single cell using transformer.',
    'long_description': '# scGPT\n\nThis is the official codebase for **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI**.\n\n[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2023.04.30.538439) &nbsp;\n[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://scgpt.readthedocs.io/en/latest/) &nbsp;\n[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE) &nbsp;\n[![PyPI version](https://badge.fury.io/py/scgpt.svg)](https://badge.fury.io/py/scgpt)\n\n**!UPDATE**: We have released several new pretrained scGPT checkpoints. Please see the [Pretrained scGPT checkpoints](#pretrained-scGPT-checkpoints) section for more details.\n\n**[2023.11.07]** As requested by many, now we have made flash-attention an optional dependency. The pretrained weights can be loaded on pytorch CPU, GPU, and flash-attn backends using the same [load_pretrained](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/utils/util.py#L304) function, `load_pretrained(target_model, torch.load("path_to_ckpt.pt"))`. An example usage is also [here](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/tasks/cell_emb.py#L258).\n\n**[2023.09.05]** We have release a new feature for reference mapping samples to a custom reference dataset or to all the millions of cells collected from CellXGene! With the help of the [faiss](https://github.com/facebookresearch/faiss) library, we achieved a great time and memory efficiency. The index of over 33 millions cells only takes less than 1GB of memory and the similarity search takes less than **1 second for 10,000 query cells on GPU**. Please see the [Reference mapping tutorial](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Reference_Mapping.ipynb) for more details.\n\n### Online apps\n\nscGPT is now available at the following online apps as well, so you can get started simply with your browser!\n\n- Run the [reference mapping app](https://app.superbio.ai/apps/299?id=6548f339a9ed6f6e5560b07d), [cell annotation app](https://app.superbio.ai/apps/274?id=64d205cb980ff714de831ee0) and the [GRN inference app](https://app.superbio.ai/apps/270?id=64b804fb823bc93b64c10a76) with cloud gpus. Thanks to the [Superbio.ai](https://app.superbio.ai/) team for helping create and host the interactive tools.\n\n## Installation\n\nscGPT works with Python >= 3.7.13 and R >=3.6.1. Please make sure you have the correct version of Python and R installed pre-installation.\n\nscGPT is available on PyPI. To install scGPT, run the following command:\n\n```bash\npip install scgpt "flash-attn<1.0.5"  # optional, recommended\n# As of 2023.09, pip install may not run with new versions of the google orbax package, if you encounter related issues, please use the following command instead:\n# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"\n```\n\n[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.\n\n```bash\npip install wandb\n```\n\nFor developing, we are using the [Poetry](https://python-poetry.org/) package manager. To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).\n\n```bash\n$ git clone this-repo-url\n$ cd scGPT\n$ poetry install\n```\n\n**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.\n\n## Pretrained scGPT Model Zoo\n\nHere is the list of pretrained models. Please find the links for downloading the checkpoint folders. We recommend using the `whole-human` model for most applications by default. If your fine-tuning dataset shares similar cell type context with the training data of the organ-specific models, these models can usually demonstrate competitive performance as well. A paired vocabulary file mapping gene names to ids is provided in each checkpoint folder. If ENSEMBL ids are needed, please find the conversion at [gene_info.csv](https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv).\n\n| Model name                | Description                                             | Download                                                                                     |\n| :------------------------ | :------------------------------------------------------ | :------------------------------------------------------------------------------------------- |\n| whole-human (recommended) | Pretrained on 33 million normal human cells.            | [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) |\n| brain                     | Pretrained on 13.2 million brain cells.                 | [link](https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx?usp=sharing) |\n| blood                     | Pretrained on 10.3 million blood and bone marrow cells. | [link](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing) |\n| heart                     | Pretrained on 1.8 million heart cells                   | [link](https://drive.google.com/drive/folders/1GcgXrd7apn6y4Ze_iSCncskX3UsWPY2r?usp=sharing) |\n| lung                      | Pretrained on 2.1 million lung cells                    | [link](https://drive.google.com/drive/folders/16A1DJ30PT6bodt4bWLa4hpS7gbWZQFBG?usp=sharing) |\n| kidney                    | Pretrained on 814 thousand kidney cells                 | [link](https://drive.google.com/drive/folders/1S-1AR65DF120kNFpEbWCvRHPhpkGK3kK?usp=sharing) |\n| pan-cancer                | Pretrained on 5.7 million cells of various cancer types | [link](https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa?usp=sharing) |\n\n## Fine-tune scGPT for scRNA-seq integration\n\nPlease see our example code in [examples/finetune_integration.py](examples/finetune_integration.py). By default, the script assumes the scGPT checkpoint folder stored in the `examples/save` directory.\n\n## To-do-list\n\n- [x] Upload the pretrained model checkpoint\n- [x] Publish to pypi\n- [ ] Provide the pretraining code with generative attention masking\n- [ ] Finetuning examples for multi-omics integration, cell type annotation, perturbation prediction, cell generation\n- [x] Example code for Gene Regulatory Network analysis\n- [x] Documentation website with readthedocs\n- [x] Bump up to pytorch 2.0\n- [x] New pretraining on larger datasets\n- [x] Reference mapping example\n- [ ] Publish to huggingface model hub\n\n## Contributing\n\nWe greatly welcome contributions to scGPT. Please submit a pull request if you have any ideas or bug fixes. We also welcome any issues you encounter while using scGPT.\n\n## Acknowledgements\n\nWe sincerely thank the authors of following open-source projects:\n\n- [flash-attention](https://github.com/HazyResearch/flash-attention)\n- [scanpy](https://github.com/scverse/scanpy)\n- [scvi-tools](https://github.com/scverse/scvi-tools)\n- [scib](https://github.com/theislab/scib)\n- [datasets](https://github.com/huggingface/datasets)\n- [transformers](https://github.com/huggingface/transformers)\n\n## Citing scGPT\n\n```bibtex\n@article{cui2023scGPT,\ntitle={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},\nauthor={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},\njournal={bioRxiv},\nyear={2023},\npublisher={Cold Spring Harbor Laboratory}\n}\n```\n',
    'author': 'Haotian',
    'author_email': 'subercui@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/bowang-lab/scGPT',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.12,<3.11',
}


setup(**setup_kwargs)
