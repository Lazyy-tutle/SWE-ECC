import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import matplotlib.pyplot as plt
import time

# Function to shuffle pixels
def shuffle_pixels(image_array, seed):
    np.random.seed(seed)
    indices = np.arange(image_array.size)
    np.random.shuffle(indices)
    return image_array.flatten()[indices].reshape(image_array.shape), indices

# Function to unshuffle pixels
def unshuffle_pixels(shuffled_array, indices):
    original_array = np.zeros_like(shuffled_array.flatten())
    original_array[indices] = shuffled_array.flatten()
    return original_array.reshape(shuffled_array.shape)

def calculate_entropy(image_array):
    # Convert the image array to grayscale
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2)
    
    # Compute histogram
    hist, _ = np.histogram(image_array.flatten(), bins=256, range=(0,256))
    
    # Calculate probability of each intensity value occurring
    probabilities = hist / np.sum(hist)
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10)) # Adding a small value to prevent log(0)
    
    return entropy

# Encrypt an image using ECC and pixel shuffling
def encrypt_image(image_path, private_key):
    # Load image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Encrypt the seed using ECC
    public_key = private_key.public_key()
    shared_key = private_key.exchange(ec.ECDH(), public_key)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'', iterations=100000, backend=default_backend())
    key = kdf.derive(shared_key)


    start_time = time.time()
    # Shuffle pixels
    seed = np.random.randint(0, 2**31 - 1)
    shuffled_array, indices = shuffle_pixels(image_array, seed)
    
    entropy_shuffle = calculate_entropy(shuffled_array)

    # Convert shuffled image to bytes
    shuffled_bytes = shuffled_array.tobytes()
    
    
    
    # Encrypt the image bytes
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_image = encryptor.update(shuffled_bytes) + encryptor.finalize()
    end_time = time.time()

    entropy_shuffle_encrypt = calculate_entropy(np.frombuffer(encrypted_image, dtype=np.uint8).reshape(image_array.shape))
    
    encryption_time = end_time - start_time

    return encrypted_image, indices, iv, public_key, encryption_time, shuffled_array, entropy_shuffle, entropy_shuffle_encrypt
# Decrypt an image using ECC and pixel shuffling
def decrypt_image(encrypted_image, indices, iv, private_key, image_shape):
    # Derive the shared key
    shared_key = private_key.exchange(ec.ECDH(), private_key.public_key())
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'', iterations=100000, backend=default_backend())
    key = kdf.derive(shared_key)
    
    # Decrypt the image bytes
    start_time = time.time()
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_bytes = decryptor.update(encrypted_image) + decryptor.finalize()

    # Convert bytes back to image array
    decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
    decrypted_array = decrypted_array.reshape(image_shape)

    # Unshuffle pixels
    unshuffled_array = unshuffle_pixels(decrypted_array, indices)
    end_time = time.time()

    decryption_time = end_time - start_time


    return unshuffled_array, decryption_time

# Function to calculate Mean Squared Error (MSE)
def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Function to calculate Peak Signal-to-Noise Ratio (PSNR)
def psnr(image1, image2):
    mse_val = mse(image1, image2)
    if mse_val == 0:
        return float('inf')  # Return infinity for perfect reconstruction
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_val))

# Generate ECC private key
private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

# Encrypt the image
original_image = Image.open('image.png')
image_shape = np.array(original_image).shape
encrypted_image, indices, iv, public_key, encryption_time, encrypted_array, entropy_shuffle, entropy_shuffle_encrypt = encrypt_image('image.png', private_key)

# Load the original image to get its shape
original_image = Image.open('image.png')
image_shape = np.array(original_image).shape

# Decrypt the image
decrypted_array, decryption_time = decrypt_image(encrypted_image, indices, iv, private_key, image_shape)

# Save the decrypted image
decrypted_image = Image.fromarray(decrypted_array)
decrypted_image.save('decrypted_image.png')

# Load images for display
encrypted_image_pil = Image.fromarray(np.frombuffer(encrypted_image, dtype=np.uint8).reshape(original_image.size[1], original_image.size[0], -1))
decrypted_image = Image.open('decrypted_image.png')

# Display images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(encrypted_image_pil)
axes[1].set_title('Encrypted Image')
axes[1].axis('off')

axes[2].imshow(decrypted_image)
axes[2].set_title('Decrypted Image')
axes[2].axis('off')

plt.show()

# Calculate PSNR and MSE

psnr_quality = psnr(np.array(original_image), decrypted_array)
mse_quality = mse(np.array(original_image), decrypted_array)

psnr_security = psnr(np.array(original_image), encrypted_array)
mse_security = mse(np.array(original_image), encrypted_array)

entropy_original = calculate_entropy(np.array(original_image))




# Print PSNR-Quality, MSE-Quality, PSNR-Security, MSE-Security, encryption time, decryption time
print("PSNR-Quality:", psnr_quality)
print("MSE-Quality:", mse_quality)
print("Encryption Time-Performance:", encryption_time)
print("Decryption Time-Performance:", decryption_time)
print("PSNR-Security:", psnr_security)
print("MSE-Security:", mse_security)
print("Entropy Original:", entropy_original)
print("Entropy Shuffle:", entropy_shuffle)
print("Entropy Shuffle Encrypt:", entropy_shuffle_encrypt)
