# 🖼️ CIFAR-10 Image Classification System (Claude-Code)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete end-to-end deep learning project for image classification using PyTorch. Train CNN models on CIFAR-10 dataset and deploy them through a professional web interface built with Streamlit. Also it's being assisted by Claude-Code

![画面収録 2025-06-13 14 21 58 (2)](https://github.com/user-attachments/assets/208167b9-697e-46d0-9687-e52c2dbfb31f)

## 🎯 Project Goals

- **Learn PyTorch fundamentals** through hands-on CNN implementation
- **Build production-ready ML pipelines** with proper project structure
- **Compare different architectures** (Simple CNN vs ResNet vs Advanced CNN)
- **Create user-friendly interfaces** for model deployment
- **Follow best practices** in ML project organization and documentation

## ✨ Key Features

### 🧠 **Multiple CNN Architectures**
- **Simple CNN**: 3-layer baseline model (95K parameters, ~68% accuracy)
- **ResNet**: Residual connections for deeper networks (272K parameters, ~75% accuracy)
- **Advanced CNN**: Custom architecture with modern techniques (2.8M parameters, ~80% accuracy)

### 🚀 **Complete Training Pipeline**
- Automated CIFAR-10 dataset downloading and preprocessing
- Data augmentation with torchvision transforms
- Real-time training progress visualization
- Model checkpointing and early stopping
- Comprehensive evaluation metrics and confusion matrices

### 🌐 **Professional Web Interface**
- **Drag & drop image upload** with real-time preprocessing
- **Interactive confidence charts** using Plotly
- **Multiple model comparison** with live switching
- **Responsive design** that works on desktop and mobile
- **Error handling** with helpful troubleshooting tips

### 📊 **Advanced Visualizations**
- Training progress plots (loss, accuracy, learning rate)
- Confusion matrices and per-class accuracy
- Model architecture comparisons
- Sample dataset visualizations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- 8GB+ RAM recommended
- 2GB free disk space for dataset and models

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/cv-image-classifier.git
cd cv-image-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Training (10 epochs)
```bash
python src/training/train_cifar10_clean.py
```

### 3. Launch Web App
```bash
# Option 1: Direct launch
streamlit run streamlit_app.py

# Option 2: Use automated launcher
python run_app.py
```

### 4. Start Classifying! 🎉
- Open browser to `http://localhost:8501`
- Upload any image and see real-time predictions
- Try different models and compare results

## 📁 Project Structure

```
cv-image-classifier/
├── 📁 src/                          # Source code modules
│   ├── 📁 data/                     # Data loading and preprocessing
│   │   ├── dataset.py               # Generic dataset utilities
│   │   ├── cifar10_dataset.py       # CIFAR-10 specific loader
│   │   └── transforms.py            # Image augmentation pipeline
│   ├── 📁 models/                   # Neural network architectures
│   │   ├── models.py                # Generic model factory
│   │   ├── cifar10_cnn.py           # CIFAR-10 specific CNNs
│   │   └── resnet.py                # Custom ResNet implementation
│   ├── 📁 training/                 # Training and evaluation scripts
│   │   ├── train.py                 # Generic training loop
│   │   ├── train_cifar10_clean.py   # CIFAR-10 training script
│   │   └── evaluate.py              # Model evaluation utilities
│   └── 📁 utils/                    # Helper functions
│       ├── utils.py                 # General utilities
│       └── config.py                # Configuration management
├── 📁 scripts/                      # Utility scripts
│   ├── setup_cifar10.py             # Dataset setup
│   └── visualize_models.py          # Architecture comparisons
├── 📁 configs/                      # Configuration files
│   └── default.json                 # Default training parameters
├── 📁 checkpoints/                  # Trained model weights
├── 📁 logs/                         # Training logs
├── 📁 data/                         # Datasets (auto-downloaded)
├── streamlit_app.py                 # Web interface application
├── train_resnet.py                  # Enhanced ResNet training
├── run_app.py                       # App launcher script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🎓 Training Models

### Basic Training (Simple CNN)
```bash
# Train with default settings (10 epochs, CPU-optimized)
python src/training/train_cifar10_clean.py

# Expected results:
# - Training time: ~10 minutes (CPU)
# - Validation accuracy: ~68%
# - Model size: 95K parameters
```

### Advanced Training (ResNet)
```bash
# Train ResNet for higher accuracy (50 epochs)
python train_resnet.py

# Expected results:
# - Training time: ~1-2 hours (CPU)
# - Validation accuracy: ~75-80%
# - Model size: 272K parameters
```

### Training Configuration
Customize training by editing configs:
```json
{
  "model_type": "resnet",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.01,
  "weight_decay": 5e-4
}
```

## 🌐 Using the Web App

### Features Overview
1. **Model Selection**: Choose between Simple CNN, ResNet, or Advanced CNN
2. **Image Upload**: Drag & drop or browse for images (PNG, JPG, JPEG)
3. **Real-time Inference**: Instant predictions with confidence scores
4. **Interactive Visualizations**: Bar charts and detailed results tables
5. **Settings Panel**: Adjust confidence thresholds and number of predictions

### Supported Image Formats
- PNG (.png)
- JPEG (.jpg, .jpeg) 
- Maximum file size: 200MB
- Recommended: Square images work best

### Sample Usage
```bash
# Start the app
streamlit run streamlit_app.py

# Upload test images:
# - Photos from your phone
# - Internet images
# - CIFAR-10 test samples
```

## 📊 Model Performance

| Model | Parameters | Accuracy | Training Time | Use Case |
|-------|------------|----------|---------------|----------|
| Simple CNN | 95K | ~68% | 10 min | Learning, prototyping |
| CIFAR-10 ResNet | 272K | ~75% | 1-2 hours | Balanced performance |
| Advanced CNN | 2.8M | ~80% | 2-4 hours | Maximum accuracy |

### Sample Results

**Simple CNN Training Progress:**
- Epoch 1: 35% → Epoch 10: 68% validation accuracy
- Smooth learning curve with no overfitting
- Fast convergence suitable for experimentation

**ResNet Performance:**
- Better feature learning with residual connections
- Higher final accuracy with efficient parameter usage
- Stable training with cosine annealing scheduler

## 🛠️ Technologies Used

### Core Frameworks
- **[PyTorch](https://pytorch.org)** - Deep learning framework
- **[torchvision](https://pytorch.org/vision/)** - Computer vision utilities
- **[Streamlit](https://streamlit.io)** - Web application framework

### Data & Visualization
- **[NumPy](https://numpy.org)** - Numerical computing
- **[Pandas](https://pandas.pydata.org)** - Data manipulation
- **[Matplotlib](https://matplotlib.org)** - Static plotting
- **[Plotly](https://plotly.com)** - Interactive visualizations
- **[Pillow](https://pillow.readthedocs.io)** - Image processing

### Development Tools
- **[tqdm](https://tqdm.github.io)** - Progress bars
- **[scikit-learn](https://scikit-learn.org)** - Evaluation metrics
- **Git** - Version control
- **Python 3.8+** - Programming language

### Architecture Highlights
- **Modular Design**: Separate modules for data, models, training
- **Configuration Management**: JSON-based parameter control
- **Caching**: Streamlit caching for fast model loading
- **Error Handling**: Comprehensive exception management
- **Responsive UI**: Mobile-friendly web interface

## 🔧 Development Setup

### Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/cv-image-classifier.git
cd cv-image-classifier

# Setup environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Run tests (if available)
pytest tests/

# Start development server
streamlit run streamlit_app.py --server.runOnSave true
```

### Adding New Models
1. Implement model in `src/models/`
2. Add to `create_model()` factory function
3. Update Streamlit app model selection
4. Test with training pipeline

### Custom Datasets
1. Create dataset loader in `src/data/`
2. Modify class names and descriptions
3. Update model output dimensions
4. Retrain with new data

## 🚀 Future Improvements

### 📈 **Model Enhancements**
- [ ] **Vision Transformers (ViT)** - Implement attention-based architectures
- [ ] **EfficientNet family** - Add EfficientNet-B0 through B7 variants
- [ ] **Data augmentation improvements** - MixUp, CutMix, AutoAugment
- [ ] **Transfer learning** - Pre-trained ImageNet models fine-tuning
- [ ] **Model ensemble** - Combine multiple models for better accuracy

### 🌐 **Web Interface Upgrades**
- [ ] **Batch processing** - Upload and classify multiple images
- [ ] **Webcam integration** - Real-time classification from camera
- [ ] **Model comparison** - Side-by-side predictions from different models
- [ ] **Gradients visualization** - Show what the model is looking at (Grad-CAM)
- [ ] **Export functionality** - Download predictions as CSV/JSON

### 🔧 **Technical Improvements**
- [ ] **GPU optimization** - CUDA acceleration and mixed precision training
- [ ] **Docker deployment** - Containerized application for easy deployment
- [ ] **API endpoint** - REST API for programmatic access
- [ ] **Model compression** - Quantization and pruning for mobile deployment
- [ ] **MLOps integration** - Weights & Biases, MLflow for experiment tracking

### 📊 **Analysis & Monitoring**
- [ ] **A/B testing framework** - Compare model performance systematically
- [ ] **Performance monitoring** - Track model drift and accuracy over time
- [ ] **Explainable AI** - SHAP values and model interpretability
- [ ] **Automated testing** - Unit tests for models and data pipeline
- [ ] **Continuous integration** - GitHub Actions for automated testing

### 🎯 **New Features**
- [ ] **Custom dataset support** - Easy integration of new image datasets
- [ ] **Transfer learning wizard** - GUI for fine-tuning pre-trained models
- [ ] **Model architecture search** - Automated neural architecture search
- [ ] **Active learning** - Smart sample selection for labeling
- [ ] **Federated learning** - Distributed training capabilities

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🐛 **Bug Reports**
- Use GitHub Issues with detailed reproduction steps
- Include system information and error messages
- Provide sample images if related to inference

### ✨ **Feature Requests**
- Open GitHub Issues with clear feature descriptions
- Explain use cases and expected benefits
- Consider implementation complexity

### 🔧 **Pull Requests**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Update documentation as needed
5. Submit a pull request with clear description

### 📝 **Development Guidelines**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write tests for new functionality
- Update README for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** - For the excellent deep learning framework
- **Streamlit Team** - For making web app development so simple
- **CIFAR-10 Dataset** - Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **ResNet Paper** - "Deep Residual Learning for Image Recognition" by He et al.
- **Open Source Community** - For countless tutorials and code examples

## 📞 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs via [GitHub Issues](https://github.com/yourusername/cv-image-classifier/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/yourusername/cv-image-classifier/discussions)
- **Email**: [your.email@example.com](mailto:your.email@example.com)

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/cv-image-classifier&type=Date)](https://star-history.com/#yourusername/cv-image-classifier&Date)

---

**Built with ❤️ for the machine learning community**

Happy coding! 🚀
