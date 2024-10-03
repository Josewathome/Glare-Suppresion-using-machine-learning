from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import gaussian_filter

# Function to display images
def display_image(title, image, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the image using Pillow
image_path = r'C:\Users\Administrator\Desktop\海面太阳耀斑图\海面太阳耀斑图\偏振海面太阳耀斑\fusedimage.bmp'

try:
    img = Image.open(image_path)
    img_rgb = img.convert('RGB')  # Ensure it is in RGB format
    img_gray = img.convert('L')  # Convert to grayscale for further processing
    img_rgb_array = np.array(img_rgb)  # Convert to NumPy array for display
    img_gray_array = np.array(img_gray)  # Grayscale NumPy array for edge detection
except FileNotFoundError:
    print(f"Error: Could not find the image at path {image_path}.")
except Exception as e:
    print(f"Error: Could not read the image. {str(e)}")
else:
    # Step 1: Display the original image
    display_image('Original Image', img_rgb_array, cmap='viridis')

    # Step 2: Display the grayscale image
    display_image('Grayscale Image', img_gray_array)

    # Step 3: Apply Edge Detection Methods

    # Apply Roberts edge detection
    edges_roberts = filters.roberts(img_gray_array)
    display_image('Roberts Edge Detection', edges_roberts)

    # Apply Sobel edge detection
    edges_sobel = filters.sobel(img_gray_array)
    display_image('Sobel Edge Detection', edges_sobel)

    # Apply Prewitt edge detection
    edges_prewitt = filters.prewitt(img_gray_array)
    display_image('Prewitt Edge Detection', edges_prewitt)

    # Apply LOG (Laplacian of Gaussian) edge detection
    edges_log = filters.laplace(img_gray_array)
    display_image('LOG Edge Detection', edges_log)

    # Apply Canny edge detection (via NumPy to simulate)
    from skimage.feature import canny
    edges_canny = canny(img_gray_array, sigma=1.0)
    display_image('Canny Edge Detection', edges_canny)

    # Step 4: Combine Edge Detection Results
    # Combine all the edge detection results to identify glare regions
    combined_edges = np.maximum.reduce([edges_roberts, edges_sobel, edges_prewitt, edges_log, edges_canny])
    display_image('Combined Edge Detection for Glare Region', combined_edges)

    # Step 5: Glare Suppression (basic simulation)
    # For simplicity, let's apply Gaussian blurring on the combined edges to suppress glare
    glare_suppressed = gaussian_filter(combined_edges, sigma=1)
    display_image('Glare-Suppressed Image', glare_suppressed)

    # Step 6: Recombine the suppressed edges with the original grayscale image
    # This step simulates the process of reducing glare and restoring image details
    reconstructed_image = img_gray_array - (glare_suppressed * 255).astype(np.uint8)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    display_image('Final Glare-Suppressed Image', reconstructed_image)

    # Step 7: Optionally, display all steps together in one plot
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    axs[0, 0].imshow(img_rgb_array)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img_gray_array, cmap='gray')
    axs[0, 1].set_title('Grayscale Image')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(combined_edges, cmap='gray')
    axs[1, 0].set_title('Combined Edge Detection')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(glare_suppressed, cmap='gray')
    axs[1, 1].set_title('Glare Suppressed (Blurring)')
    axs[1, 1].axis('off')

    axs[2, 0].imshow(reconstructed_image, cmap='gray')
    axs[2, 0].set_title('Final Reconstructed Image')
    axs[2, 0].axis('off')

    plt.tight_layout()
    plt.show()
