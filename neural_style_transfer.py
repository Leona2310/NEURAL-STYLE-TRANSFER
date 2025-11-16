# neural_style_transfer.py
# ------------------------
# Neural Style Transfer Script
# Combines a content image (photo) and a style image (artwork)
# to create a stylized version of the content image.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Step 1: Load and preprocess images ----------
def load_image(img_path, max_size=400):
    """Load and resize image, convert to tensor."""
    image = Image.open(img_path).convert('RGB')
    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image

# Replace these with your actual filenames
content = load_image("my_photo.jpg")        # Your photo
style = load_image("starry_night.jpg")      # Painting or art style

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content = content.to(device)
style = style.to(device)

# ---------- Step 2: Load pretrained VGG19 model ----------
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Layers for content and style representation
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# ---------- Step 3: Helper functions ----------
def get_features(image, model):
    """Extract features from specific layers of the VGG19 model."""
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """Compute the Gram matrix (used to capture style)."""
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram 

# ---------- Step 4: Extract features ----------
with torch.no_grad():
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# ---------- Step 5: Create target image ----------
target = content.clone().requires_grad_(True).to(device)

# ---------- Step 6: Define optimizer and weights ----------
style_weight = 1e6
content_weight = 1
optimizer = optim.Adam([target], lr=0.003)

# ---------- Step 7: Run style transfer ----------
steps = 200  # reduce to 500 for faster result

print("Starting style transfer... This may take a few minutes.\n")

for step in range(1, steps + 1):
    target_features = get_features(target, vgg)
    
    # Detach content and style references from graph
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'].detach()) ** 2)
    
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer].detach()
        layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}/{steps}, Total loss: {total_loss.item():.4f}")

# ---------- Step 8: Show and save the final image ----------
final_img = target.clone().detach().cpu().squeeze()
final_img = transforms.ToPILImage()(final_img)
final_img.save("styled_image.jpg")

print("\n✅ Style transfer complete! Saved as styled_image.jpg")

plt.imshow(final_img)
plt.title("Styled Image")
plt.axis('off')
plt.show()
