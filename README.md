# RadMultiBench Model Reproduction

This repository contains the code to reproduce the experiments for the paper "A Multimodal Benchmark Dataset for Diagnostic Radiology."
The code is structured in a modular, config-driven way for reproducibility.

## 1. Setup

### Prerequisites
- Python 3.8+
- PyTorch
- [cite_start]An NVIDIA GPU (T4 recommended, as used in paper [cite: 190])

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/RadMultiBench_Reproduce.git](https://github.com/your-username/RadMultiBench_Reproduce.git)
    cd RadMultiBench_Reproduce
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `hi-ml-multimodal` and `clip` packages may take a moment to install.*

3.  Download the data:
    ```bash
    # (Add your !gdown command here)
    !gdown '[https://docs.google.com/uc?export=download&id=1JIOb4K-pmTxdXJhCjU5okpO6rrNiTuYf](https://docs.google.com/uc?export=download&id=1JIOb4K-pmTxdXJhCjU5okpO6rrNiTuYf)' -O data.zip
    unzip data.zip
    rm data.zip
    ```
4.  Verify your data paths in the config files (`configs/*.yaml`). The default path assumes the `Medical/` directory is in the root.
    ```yaml
    DATA:
      IMG_DIR: "./Medical/images"
      CSV_PATH: "./Medical/clean_map.csv"
    ```

## 2. Training

The main entry point is `train.py`. You must specify a configuration file using `--config-file`.

### Example Commands:

**To train ResNet-LSTM (Classification):**
```bash
python train.py --config-file configs/model_resnet_lstm_class.yaml
