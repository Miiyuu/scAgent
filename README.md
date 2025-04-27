# scAgent

Universal Cell Type Annotation via a Large Language Model (LLM)

## Test Model and Data

Model checkpoints and required data files can be downloaded from Zenodo:

- Zenodo DOI: [10.5281/zenodo.1234567](https://zenodo.org/record/1234567) (Miiyuu/scAgent: Universal Cell Type Annotation via a LLM)

Note: Files and directories marked with `*` in the project structure need to be downloaded from Zenodo and placed in their corresponding directories.

## Project Structure

```text
.
├── agent/                          # Core agent modules for cell type annotation
│   ├── cell_searcher.py            # Searches for similar cells in the database
│   ├── ft_increment.py             # Handles fine-tuning and incremental learning
│   ├── input_analyzer.py           # Processes and analyzes input queries
│   ├── novel_assess.py             # Assesses novel cell type discoveries
│   ├── novel_detection/            # Novel cell type detection modules
│   │   ├── calculate_dis.py        # Calculates distances between cell embeddings
│   │   ├── __init__.py            # Package initialization
│   │   ├── LLModel.py             # Large language model integration
│   │   ├── make_gene_idx.py       # Creates gene indices for analysis
│   │   ├── openset_identify.py    # Identifies unknown cell types
│   │   └── process_data.py        # Processes single-cell data
│   ├── output_generator.py         # Generates final annotation outputs
│   └── tissue_searcher.py          # Searches for tissue-specific information
│
├── cg-loras*/                      # LoRA (Low-Rank Adaptation) model files (from Zenodo)
│
├── dict/                           # Dictionary and mapping files
│   ├── cell_id2name.json           # Mapping between cell IDs and names
│   ├── center_lora.pkl             # Centroid data for LoRA models
│   ├── center_nolora.pkl           # Centroid data for non-LoRA models
│   ├── id2type.json                # Mapping between IDs and cell types
│   ├── tissue_general_mapping.json # General tissue type mappings
│   ├── tissue_id2name.json         # Mapping between tissue IDs and names
│   └── unique_cell_types.csv       # List of unique cell types
│
├── flash-attention-main/           # Flash attention implementation
│
├── LoRA-main/                      # LoRA model implementation
│
├── model*/                         # Pretrained models (from Zenodo)
│   ├── cls_params_200.pth          # Classification parameters
│   ├── cls_params_tissue.pth       # Tissue classification parameters
│   ├── pretrained_models/          # Pretrained model files
│   │   ├── args.json               # Model arguments
│   │   ├── best_model.pt           # Best performing model
│   │   └── vocab.json              # Vocabulary file
│   ├── tissue_lora.pt              # Tissue-specific LoRA model
│   └── tissue_lora_ts.pt           # Timestamped tissue LoRA model
│
├── scGPT-0.2.1/                    # scGPT model implementation
│
├── config.py                       # Configuration settings
│
├── graph.py                        # Visualization and graph utilities
│
├── main.py                         # Main execution script
│
├── memory/                         # Memory and database files
│   ├── cell_embedding2.db*          # Cell embedding database (from Zenodo)
│   ├── cell_embedding.db*           # Primary cell embedding database (from Zenodo)
│   └── search_db.py                # Database search utilities
│
├── output.png                      # Example output visualization
│
├── test*/                          # Test datasets (from Zenodo)
│   ├── cancer_data.h5ad            # Cancer dataset example
│   ├── new_adata.h5ad              # General test dataset
│   └── tissue15_1.h5ad             # Tissue-specific test dataset
│
└── tool/                           # Utility tools
    ├── __init__.py                 # Package initialization
    ├── build_model.py              # Model building utilities
    ├── data_process.py             # Data processing utilities
    └── scgpt_model.py              # scGPT model utilities
```

## Installation

### via pip

1. First create a conda environment:
```bash
conda create -n scagent python=3.10
conda activate scagent
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Install required packages from subdirectories:
```bash
cd flash-attention-main && pip install -e . && cd ..
cd LoRA-main && pip install -e . && cd ..
cd scGPT-0.2.1 && pip install -e . && cd ..
```

## System Requirements

- **Operating system**: Ubuntu 20.04.3 LTS (used OS version)
- **Python version**: 3.10
- **Packages**: See requirements.txt

## Hardware Requirements

- GPU with at least 8GB VRAM (recommended)
- Tested on NVIDIA GeForce RTX 4090 with CUDA Version: 12.2

## Run

1. Set up the project structure as described above
2. Modify necessary directories and your questions in `config.py`
3. Run the main script:
```bash
python main.py
```

This will generate answers to your example questions.

## License

Apache-2.0 license

## Authors

*Corresponding author(s): gaoyj@zju.edu.cn  
Contributing authors: yuren.mao@zju.edu.cn; miyu@zju.edu.cn; peigenliu@zju.edu.cn; zmengfei@zju.edu.cn; hanqingliu@g.harvard.edu

## Citation

If you use scAgent in your research, please cite:

```bibtex
@article{mao2025scagent,
  title={scAgent: Universal Single-Cell Annotation via a LLM Agent},
  author={Mao, Yuren and Mi, Yu and Liu, Peigen and Zhang, Mengfei and Liu, Hanqing and Gao, Yunjun},
  journal={arXiv preprint arXiv:2504.04698},
  year={2025}
}
```
