import os
import cv2
import numpy as np
import string

# Where to save the dataset
dataset_dir = r"D:\bml\braille_dataset"
os.makedirs(dataset_dir, exist_ok=True)  # ✅ Create main folder

# Braille patterns for A-Z
braille_patterns = {
    'A': [(0,0)],
    'B': [(0,0),(1,0)],
    'C': [(0,0),(0,1)],
    'D': [(0,0),(0,1),(1,1)],
    'E': [(0,0),(1,1)],
    'F': [(0,0),(1,0),(0,1)],
    'G': [(0,0),(1,0),(0,1),(1,1)],
    'H': [(0,0),(1,0),(1,1)],
    'I': [(1,0),(0,1)],
    'J': [(1,0),(0,1),(1,1)],
    'K': [(0,0),(2,0)],
    'L': [(0,0),(1,0),(2,0)],
    'M': [(0,0),(0,1),(2,0)],
    'N': [(0,0),(0,1),(1,1),(2,0)],
    'O': [(0,0),(1,1),(2,0)],
    'P': [(0,0),(1,0),(0,1),(2,0)],
    'Q': [(0,0),(1,0),(0,1),(1,1),(2,0)],
    'R': [(0,0),(1,0),(1,1),(2,0)],
    'S': [(1,0),(0,1),(2,0)],
    'T': [(1,0),(0,1),(1,1),(2,0)],
    'U': [(0,0),(2,0),(2,1)],
    'V': [(0,0),(1,0),(2,0),(2,1)],
    'W': [(1,0),(0,1),(1,1),(2,1)],
    'X': [(0,0),(0,1),(2,0),(2,1)],
    'Y': [(0,0),(0,1),(1,1),(2,0),(2,1)],
    'Z': [(0,0),(1,1),(2,0),(2,1)]
}

# Create images for each character
img_size = 100
dot_radius = 10

for letter, pattern in braille_patterns.items():
    letter_dir = os.path.join(dataset_dir, letter)
    os.makedirs(letter_dir, exist_ok=True)  # ✅ Create subfolder for each letter
    
    for i in range(200):  # Create 200 variations
        img = np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        # Draw braille dots
        for (row, col) in pattern:
            center_x = 30 + col * 30
            center_y = 30 + row * 30
            cv2.circle(img, (center_x, center_y), dot_radius, (0,), -1)
        
        filename = os.path.join(letter_dir, f"{letter}_{i}.png")
        cv2.imwrite(filename, img)

print(f"✅ Braille dataset generated in: {dataset_dir}")
