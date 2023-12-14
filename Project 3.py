import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    return cv2.imread(image_path)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_threshold(image, threshold_value=100, max_value=255, threshold_type=cv2.THRESH_BINARY):
    _, thresholded_image = cv2.threshold(image, threshold_value, max_value, threshold_type)
    return thresholded_image

def apply_canny_edge_detection(image, threshold1=30, threshold2=100):
    return cv2.Canny(image, threshold1, threshold2)

def filter_contours(contours, min_area=1000):
    return [contour for contour in contours if cv2.contourArea(contour) > min_area]

def create_blank_mask(image):
    return np.zeros_like(image)

def draw_contours_on_mask(mask, contours):
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

def extract_image_with_mask(original_image, mask):
    return cv2.bitwise_and(original_image, original_image, mask=mask)

def display_images(original_image, extracted_image):
    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB)), plt.title('Extracted Image')
    plt.show()

# Load the image
image_path = r"C:\Users\samby\OneDrive\Documents\GitHub\AER-850-Project-3\motherboard_image.JPEG"
original_image = load_image(image_path)

# Image processing steps
gray_image = convert_to_grayscale(original_image)
thresholded_image = apply_threshold(gray_image)
edges = apply_canny_edge_detection(thresholded_image)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = filter_contours(contours)
mask = create_blank_mask(gray_image)
draw_contours_on_mask(mask, filtered_contours)
extracted_image = extract_image_with_mask(original_image, mask)

# Display the results
display_images(original_image, extracted_image)
