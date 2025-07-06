import io
import random
from PIL import Image, ImageFilter
import imagehash
import piexif
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def get_random_device_info():
    """
    Returns a dictionary with randomized device/app info for the Telethon client.
    """
    devices = [
        # iPhones
        {"device_model": "iPhone 14 Pro", "system_version": "16.5", "app_version": "9.6.3"},
        {"device_model": "iPhone 13", "system_version": "16.2", "app_version": "9.4.1"},
        # Androids
        {"device_model": "Pixel 7", "system_version": "13.0", "app_version": "9.6.7"},
        {"device_model": "Galaxy S23 Ultra", "system_version": "13.0", "app_version": "9.5.2"},
        # Desktops
        {"device_model": "PC 64bit", "system_version": "Windows 11", "app_version": "4.8.1"},
        {"device_model": "MacBook Pro", "system_version": "macOS 13.4", "app_version": "9.6.3"},
    ]
    return random.choice(devices)

def clean_image_metadata(image_bytes: bytes) -> bytes:
    """
    Removes all EXIF and other metadata from an image using piexif.
    """
    try:
        # piexif.remove can fail on images without exif, so we wrap it
        return piexif.remove(image_bytes)
    except (ValueError, piexif.InvalidImageDataError):
        # If it fails, it means no valid EXIF data was found, so we can pass
        return image_bytes
    except Exception as e:
        logger.warning(f"Could not strip metadata from image: {e}")
        return image_bytes

def distort_image(image: Image.Image) -> Image.Image:
    """
    Applies a chain of subtle, randomized distortions to an image to defeat hash-based detection.
    """
    # 1. Slight Rotation
    angle = random.uniform(-2.0, 2.0)
    distorted = image.rotate(angle, resample=Image.BICUBIC, expand=False)

    # 2. Slight Resize
    original_size = distorted.size
    scale = random.uniform(0.98, 1.02)
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    distorted = distorted.resize(new_size, Image.LANCZOS)
    distorted = distorted.resize(original_size, Image.LANCZOS)

    # 3. Tiny Crop
    left = random.randint(1, 2)
    top = random.randint(1, 2)
    right = distorted.width - random.randint(1, 2)
    bottom = distorted.height - random.randint(1, 2)
    distorted = distorted.crop((left, top, right, bottom))
    distorted = distorted.resize(original_size, Image.LANCZOS) # Resize back to original dimensions

    # 4. Re-compress with random quality
    output = io.BytesIO()
    jpeg_quality = random.randint(85, 95)
    distorted.save(output, format='JPEG', quality=jpeg_quality)
    output.seek(0)

    return Image.open(output)

def get_image_hashes(image: Image.Image) -> dict:
    """
    Calculates multiple perceptual hashes for an image.
    """
    try:
        # Convert to grayscale for hash consistency
        grayscale_image = image.convert("L")
        hashes = {
            "phash": str(imagehash.phash(grayscale_image)),
            "dhash": str(imagehash.dhash(grayscale_image)),
            "average_hash": str(imagehash.average_hash(grayscale_image)),
        }
        return hashes
    except Exception as e:
        logger.error(f"Error calculating image hashes: {e}")
        return {}

async def process_stealth_image(
    image_bytes: bytes,
    trap_hashes: list[str]
) -> Tuple[Optional[io.BytesIO], bool, str]:
    """
    Processes an image for stealth reposting:
    1. Cleans metadata.
    2. Calculates multiple hashes and checks against trap list.
    3. Applies distortion.
    4. Returns a BytesIO object ready for upload, a trap flag, and a reason.
    """
    try:
        # 1. Clean metadata first
        cleaned_bytes = clean_image_metadata(image_bytes)
        image = Image.open(io.BytesIO(cleaned_bytes))
        image = image.convert("RGB") # Standardize format

        # 2. Check hashes against trap list
        hashes = get_image_hashes(image)
        if not hashes:
            return None, True, "Failed to calculate image hashes"

        for hash_type, hash_value in hashes.items():
            if hash_value in trap_hashes:
                reason = f"Image matched a blocked {hash_type}: {hash_value}"
                return None, True, reason

        # 3. Apply distortion for stealth
        distorted_image = distort_image(image)

        # 4. Prepare for upload
        output = io.BytesIO()
        distorted_image.save(output, format='JPEG')
        output.seek(0)
        output.name = f"stealth_{random.randint(1000, 9999)}.jpg"

        return output, False, ""

    except Exception as e:
        logger.error(f"Error in process_stealth_image: {e}")
        return None, True, f"Image processing failed: {e}"