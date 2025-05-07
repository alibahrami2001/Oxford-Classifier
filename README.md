
# 🐾 Oxford Pets: Cat vs Dog Classifier

A convolutional neural network (CNN) classifier trained to distinguish between cat and dog images from the Oxford-IIIT Pet Dataset.

![Labels Distribution](https://via.placeholder.com/400?text=Class+Distribution+Chart+Here)  
*Example class distribution*



## 📁 Repository Structure

```

.
├── Oxford\_pet\_classification\_.ipynb  # Main training notebook
├── oxford\_cat\_dog.pt                 # Trained model weights
├── README.md                         # This file
└── /content                          # Dataset directory (auto-created)

````

---

## 📚 Dataset

- **Source**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Labeling Rule**:
  - Uppercase filename → Cat (`[0]`)
  - Lowercase filename → Dog (`[1]`)
- **Statistics**:
  - 7,349 total images
  - Balanced distribution (≈50% cats, ≈50% dogs)

---

## 🧠 Model Architecture

**CNN Sequence:**
```python
Sequential(
    ConvBlock(3, 8),       # Input: 256x256 RGB
    ConvBlock(8, 16),
    ConvBlock(16, 32),
    ConvBlock(32, 64),
    ConvBlock(64, 128),
    ConvBlock(128, 256),
    ConvBlock(256, 512),
    AdaptiveAvgPool2d(),
    Flatten(),
    Linear(512, 2)         # Output: [cat_prob, dog_prob]
)
````

Each `ConvBlock` contains:

* Conv2d (3x3 kernel)
* BatchNorm
* ReLU Activation
* Dropout (10%)
* MaxPool (2x2)

---

## ⚙️ Training Details

* **Device**: CUDA-enabled GPU (falls back to CPU)
* **Epochs**: 100
* **Batch Size**: 512
* **Optimizer**: Adam (`lr=3e-4`, `weight_decay=1e-5`)
* **Loss**: CrossEntropyLoss

**Checkpoints:**

* `oxford_cat_dog_best.pt`: Best validation loss
* `oxford_cat_dog_last.pt`: Final epoch weights

---

## 🚀 Usage

### Requirements

```bash
pip install torch torchvision tqdm matplotlib scikit-learn
```

### Inference Example

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("oxford_cat_dog.pt", map_location=device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Prediction
img = Image.open("your_image.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(tensor)
prediction = torch.argmax(logits).item()
print(f"Prediction: {'Cat' if prediction == 0 else 'Dog'}")
```

---

## 📊 Performance

Final evaluation metrics typically show:

* **>98% Accuracy**
* **F1-scores >0.97** for both classes
* Detailed classification report included in the notebook

---

## 📜 License

MIT License (see repository for details)

---

> **Note**: Place the `oxford_cat_dog.pt` file in your repository root. For full reproducibility, run the notebook in a GPU-enabled environment with the dataset auto-downloaded to `/content`.

