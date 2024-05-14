import streamlit as st
import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

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

# Encrypt an image using ECC and pixel shuffling
def encrypt_image(image_array, private_key):
    # Shuffle pixels
    seed = np.random.randint(0, 2**31 - 1)
    shuffled_array, indices = shuffle_pixels(image_array, seed)

    # Convert shuffled image to bytes
    shuffled_bytes = shuffled_array.tobytes()

    # Encrypt the seed using ECC
    public_key = private_key.public_key()
    shared_key = private_key.exchange(ec.ECDH(), public_key)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'', iterations=100000, backend=default_backend())
    key = kdf.derive(shared_key)
    
    # Encrypt the image bytes
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_image = encryptor.update(shuffled_bytes) + encryptor.finalize()

    return encrypted_image, indices, iv, private_key.public_key()

# Decrypt an image using ECC and pixel shuffling
def decrypt_image(encrypted_image, indices, iv, private_key, image_shape):
    # Derive the shared key
    shared_key = private_key.exchange(ec.ECDH(), private_key.public_key())
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'', iterations=100000, backend=default_backend())
    key = kdf.derive(shared_key)
    
    # Decrypt the image bytes
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_bytes = decryptor.update(encrypted_image) + decryptor.finalize()

    # Convert bytes back to image array
    decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
    decrypted_array = decrypted_array.reshape(image_shape)

    # Unshuffle pixels
    unshuffled_array = unshuffle_pixels(decrypted_array, indices)

    return unshuffled_array

# Streamlit app code
def main():
    st.sidebar.title("ECC with pixel Shuffling App")
    page = st.sidebar.selectbox("Choose a Page", ["Home", "Image Encryption"])

    if page == "Home":
        render_home_page()
    elif page == "Image Encryption":
        render_image_encryption_page()

def render_home_page():
    st.title("ECC with pixel Shuffling App")
    st.write("Select a method from the sidebar to get started.")

def render_image_encryption_page():
    st.title("Image Encryption with ECC and Pixel Shuffling")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        st.image(image, caption='Original Image', use_column_width=True)

        # Generate ECC private key
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

        # Encrypt the image
        encrypted_image, indices, iv, public_key = encrypt_image(image_array, private_key)

        # Decrypt the image
        decrypted_array = decrypt_image(encrypted_image, indices, iv, private_key, image_array.shape)

        # Convert decrypted array to image
        decrypted_image = Image.fromarray(decrypted_array)

        # Display encrypted and decrypted images
        st.subheader("Encrypted Image")
        encrypted_image_pil = Image.fromarray(np.frombuffer(encrypted_image, dtype=np.uint8).reshape(image_array.shape))
        st.image(encrypted_image_pil, use_column_width=True)

        st.subheader("Decrypted Image")
        st.image(decrypted_image, use_column_width=True)

if __name__ == "__main__":
    main()
