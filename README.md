<div align="center">
  <img src="https://i.postimg.cc/FsjR8T5c/Logo.png"
       alt="PolypX Logo"
       height="400"
       width="700"
       style="border-radius: 12px;"/>
</div>

**PolypX** is a deep learning-based solution for **polyp segmentation in colonoscopy images**.
It leverages the **DeepLab v3+ architecture** to accurately detect and outline polyp regions — a critical step in the early detection and diagnosis of colorectal cancer.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Training](#2-training)
  - [3. Prediction](#3-prediction)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **DeepLab v3+ Architecture**: Utilizes the powerful DeepLab v3+ model for accurate and efficient semantic segmentation of polyps.
- **Data Augmentation**: Applies a variety of augmentation techniques to improve robustness and reduce overfitting.
- **Custom Data Loader**: Includes a custom data loader tailored for the **CVC-ClinicDB** dataset.
- **Comprehensive Evaluation Metrics**: Supports Dice Coefficient and Intersection over Union (IoU) for reliable performance measurement.
- **Pre-trained Weights**: Provides a ready-to-use pre-trained model located in the `checkpoint/` directory for quick inference.

## System Requirements

- Python 3.8 or higher
- Linux, macOS, or Windows
- (Optional) CUDA-enabled GPU for faster training

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/A-A7med-i/PolypX.git
   cd PolypX
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation

The model is trained on the **CVC-ClinicDB dataset**. Before running the pipeline, ensure your data is structured as follows:

```markdown
data/
└── CVC-ClinicDB/
    ├── Original/      # Original colonoscopy images
    │   ├── 1.tif
    │   └── ...
    └── Ground Truth/  # Segmentation masks
        ├── 1.tif
        └── ...
```

### 2. Training

You can train the model using the main training script from your terminal:

```bash
python -m src.pipeline.main
```

This will start the training process using the configurations defined in the source code.

### 3. Prediction

Once the model is trained, the best weights are saved to `checkpoint/best_model_weights.pth`. You can use these weights to make predictions on new images.

To perform inference, adapt the prediction logic from the `src/pipeline/pipeline.py` script.
Load the pre-trained weights and pass your images through the model to get the segmentation masks.

## Documentation

For more details about the pipeline, preprocessing, model architecture, training, and results, please refer to the `docs/` folder. It contains step-by-step markdown files explaining each part of the project.

## Project Structure

``` markdown
PolypX/
├── checkpoint/                      # Stores model checkpoints
│   └── best_model_weights.pth       # Best pre-trained weights
│
├── data/                            # Dataset and metadata
│   ├── metadata.csv                 # Dataset annotations
│   ├── README.txt                   # Dataset description
│   └── CVC-ClinicDB/                # Dataset folder
│       ├── Ground Truth/            # Segmentation masks
│       └── Original/                # Colonoscopy images
│
├── docs/                            # Documentation
│   ├── 01_pipeline_overview.md
│   ├── 02_model_architecture.md
│   ├── 03_loss.md
│   ├── 04_metrics.md
│   ├── 05_training.md
│   └── 06_pipeline_results.md
│
├── src/                             # Source code
│   ├── config/                      # Configurations
│   │   └── constant.py
│   ├── data/                        # Data handling
│   │   ├── custom_data.py
│   │   └── loader.py
│   ├── entities/                    # Entity definitions
│   │   └── models.py
│   ├── loss/                        # Loss functions
│   │   └── loss.py
│   ├── metrics/                     # Evaluation metrics
│   │   └── metrics.py
│   ├── model/                       # Model architectures
│   │   └── model.py
│   ├── pipeline/                    # Pipeline scripts
│   │   ├── main.py
│   │   └── pipeline.py
│   ├── training/                    # Training loop
│   │   └── trainer.py
│   └── utils/                       # Utility functions
│       └── helper.py
│
├── requirements.txt                 # Dependencies
├── setup.py                         # Setup script
├── README.md                        # Project documentation
├── .gitignore                       # Ignore rules
└── LICENSE                          # License
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
