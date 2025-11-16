# NEURAL-STYLE-TRANSFER

COMPANY : CODTECH IT SOLUTIONS

NAME : LEONA MENDES

INTERN ID : CT04DR1252

DOMAIN NAME : ARTIFICIAL INTELLIGENCE

DURATION : 4 WEEKS 

MENTOR : NEELA SANTOSH


# TASK 3 — NEURAL-STYLE-TRANSFER

Neural Style Transfer (NST) was the third task, and the main objective of this task was to combine two different images—one representing the content and the other representing the style—to generate a new image that keeps the structure of the original photograph while being visually transformed into the artistic style of another image. In this task, the code you wrote used deep learning techniques, specifically a pretrained VGG19 convolutional neural network, to analyze and extract patterns from images. The idea behind NST is based on the discovery that different layers of CNNs capture different types of information: shallow layers capture texture and style patterns, while deeper layers capture the meaningful content and structure of an image. Your script begins by loading and preprocessing two images: a content image (my_photo.jpg) and a style image (starry_night.jpg). The preprocessing step resizes the images, converts them to tensors, and prepares them for the neural network. After this, the VGG19 model is loaded in evaluation mode, meaning it won't train or change but will simply extract the visual features needed for content and style comparison. You specifically selected certain layers of the VGG19 model—conv4_2 for content and layers like conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 for style extraction—because these layers have been proven to represent meaningful visual information for NST.

Next, your code uses a function called Gram Matrix, which is essential for representing the style of an image. The Gram Matrix captures texture patterns by measuring how different feature maps from the neural network relate to each other. After extracting the feature representations of both images, the model compares the target image (initially a clone of the content image) to both the content and style features. The goal of optimization is to adjust the target image so that it minimizes two losses: content loss, which ensures the image does not lose its original structure, and style loss, which ensures it visually resembles the chosen artwork. You used the Adam optimizer with a learning rate of 0.003 to iteratively update the target image over 200 steps. During each iteration, the optimizer reduces the total loss, a combination of the content and style losses, by modifying the pixel values of the target image. The style loss is given significantly more weight (1e6) compared to the content loss (1), which allows the artistic patterns and brushstroke-like textures to strongly influence the final result.

Once the optimization loop completes, the script converts the tensor back into an image and saves the generated artwork as styled_image.jpg. This final output visually blends the two sources: the photo remains recognizable, but the texture, color palette, and artistic strokes reflect the "Starry Night" style. Neural Style Transfer has real-world applications in digital art creation, film animation, photo stylization, augmented reality filters, video editing, design prototyping, and even restoring damaged art pieces by recreating stylistic textures. It has become a powerful tool in creative industries because it allows anyone—even without artistic skills—to generate stylized artwork using AI. Your task demonstrates how deep learning models like VGG19 can be repurposed for highly creative computational tasks beyond traditional classification.

# OUTPUT 

<img width="836" height="691" alt="Image" src="https://github.com/user-attachments/assets/ee849713-68a8-4a5a-aadd-a72ed74e8f0e" />
