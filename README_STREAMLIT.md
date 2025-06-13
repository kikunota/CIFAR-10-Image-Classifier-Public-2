# 🖼️ CIFAR-10 Image Classification Web App

A professional, interactive web interface for image classification using trained CNN models built with Streamlit.

![Streamlit App Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## ✨ Features

### 🎯 **Core Functionality**
- **Drag & Drop Upload**: Intuitive file upload with drag-and-drop support
- **Real-time Inference**: Instant predictions using trained CNN models
- **Interactive Visualizations**: Dynamic confidence charts with Plotly
- **Multiple Model Support**: Choose from Simple CNN, Advanced CNN, or ResNet
- **Professional UI**: Clean, modern interface with custom styling

### 📊 **Advanced Features**
- **Confidence Filtering**: Adjustable confidence thresholds
- **Top-K Predictions**: Configurable number of predictions to display
- **Error Handling**: Comprehensive error messages and troubleshooting tips
- **Model Information**: Real-time model stats and architecture details
- **Responsive Design**: Works on desktop and mobile devices

### 🎨 **User Experience**
- **Tabbed Interface**: Organized sections for upload, samples, and information
- **Progress Indicators**: Loading spinners for processing feedback
- **Color-coded Results**: Visual confidence indicators (green/yellow/red)
- **Detailed Descriptions**: Class descriptions and model explanations

## 🚀 Quick Start

### 1. **Prerequisites**
Make sure you have a trained model from the training pipeline:
```bash
python src/training/train_cifar10_clean.py
```

### 2. **Install Dependencies**
```bash
pip install streamlit plotly pandas
```

### 3. **Run the App**
```bash
streamlit run streamlit_app.py
```

### 4. **Open in Browser**
The app will automatically open in your default browser at:
```
http://localhost:8501
```

## 📱 Using the App

### **Upload & Classify Tab**
1. **Choose Model**: Select CNN architecture in the sidebar
2. **Upload Image**: Drag & drop or click to browse for image files
3. **Adjust Settings**: Set confidence threshold and number of predictions
4. **View Results**: See predictions with confidence scores and visualizations

### **Sample Images Tab**
- Test with provided sample images
- Quick way to try different classes
- No need to find your own test images

### **About Tab**
- Learn about the model architecture
- Understand how the classification works
- View technical details and performance metrics

## 🎛️ App Configuration

### **Sidebar Controls**
- **Model Selection**: Choose between Simple CNN, Advanced CNN, or ResNet
- **Confidence Threshold**: Filter predictions below certain confidence level
- **Top K Predictions**: Number of predictions to display (3-10)

### **Supported File Formats**
- PNG (.png)
- JPEG (.jpg, .jpeg)
- Maximum file size: 200MB

## 🏗️ Technical Architecture

### **Frontend (Streamlit)**
```python
streamlit_app.py
├── UI Components
│   ├── Header & Navigation
│   ├── Sidebar Controls
│   ├── File Upload Widget
│   ├── Results Display
│   └── Interactive Charts
├── Processing Pipeline
│   ├── Image Preprocessing
│   ├── Model Inference
│   └── Results Formatting
└── Styling & Layout
    ├── Custom CSS
    ├── Responsive Design
    └── Error Handling
```

### **Backend Integration**
- **Model Loading**: Cached PyTorch model loading
- **Image Processing**: PIL + torchvision transforms
- **Inference Engine**: Real-time CNN predictions
- **Visualization**: Plotly interactive charts

## 🎨 Customization

### **Styling**
The app uses custom CSS for professional appearance:
- Gradient headers
- Card-based layouts
- Color-coded confidence levels
- Responsive grid system

### **Adding New Models**
To add a new model architecture:
1. Add model type to `model_type` selectbox
2. Update `load_model()` function
3. Add model info to sidebar display

### **Custom Classes**
To use with different datasets:
1. Update `CIFAR10_CLASSES` list
2. Modify `CLASS_DESCRIPTIONS` dictionary
3. Adjust model loading for different class counts

## 🔧 Troubleshooting

### **Common Issues**

**App won't start:**
```bash
# Check if streamlit is installed
python -m streamlit --version

# Install if missing
pip install streamlit
```

**Model not found:**
```bash
# Ensure model exists
ls checkpoints/cifar10_simple_cnn/model_best.pth

# Train model if missing
python src/training/train_cifar10_clean.py
```

**Import errors:**
```bash
# Run from project root directory
cd cv-image-classifier
streamlit run streamlit_app.py
```

**Image upload fails:**
- Check file format (PNG, JPG, JPEG only)
- Ensure file size < 200MB
- Try with a different image
- Check browser console for errors

### **Performance Tips**
- Use smaller images for faster processing
- Close other browser tabs if experiencing slowdowns
- Restart the app if memory usage becomes high

## 📊 Model Performance

The web app displays real-time model performance metrics:

| Model Type | Parameters | Accuracy | Speed |
|------------|------------|----------|-------|
| Simple CNN | 95K | 68% | Fast |
| Advanced CNN | 2.8M | ~75% | Medium |
| ResNet | 272K | ~70% | Medium |

## 🌐 Deployment Options

### **Local Development**
```bash
streamlit run streamlit_app.py
```

### **Streamlit Cloud**
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

### **Production Considerations**
- Use `--server.port` and `--server.address` for custom hosting
- Enable `--server.enableCORS=false` for cross-origin requests
- Set `--server.enableXsrfProtection=false` if needed
- Configure `--browser.gatherUsageStats=false` for privacy

## 🛠️ Development

### **Project Structure**
```
cv-image-classifier/
├── streamlit_app.py          # Main Streamlit application
├── src/                      # Source code modules
│   ├── models/              # Model architectures
│   ├── data/                # Data processing
│   └── utils/               # Utility functions
├── checkpoints/             # Trained model weights
├── samples/                 # Sample images (optional)
└── requirements.txt         # Python dependencies
```

### **Key Functions**
- `load_model()`: Cached model loading
- `preprocess_image()`: Image preprocessing pipeline
- `make_prediction()`: Inference and result formatting
- `render_*()`: UI component rendering functions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the Streamlit app
5. Submit a pull request

## 📞 Support

- **Documentation**: Check this README and app's About tab
- **Issues**: Report bugs via GitHub Issues
- **Questions**: Use GitHub Discussions for questions

---

**Happy Classifying! 🎉**