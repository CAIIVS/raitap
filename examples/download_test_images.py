"""
Download sample test images for transparency examples.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path


def download_image(url: str, output_path: Path) -> bool:
    """Download an image from URL with proper headers."""
    try:
        # Add User-Agent to avoid 403 errors
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            output_path.write_bytes(response.read())
        print(f"✓ Downloaded: {output_path.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {output_path.name}: {e}")
        return False


def main():
    # Create output directory
    output_dir = Path("outputs/test_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample images from various sources
    # Using smaller images to avoid issues
    images = [
        {
            "url": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02099601_golden_retriever.JPEG",
            "filename": "golden_retriever.jpg",
        },
        {
            "url": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02123159_tiger_cat.JPEG",
            "filename": "tiger_cat.jpg",
        },
        {
            "url": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01440764_tench.JPEG",
            "filename": "tench.jpg",
        },
        {
            "url": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02086240_Shih-Tzu.JPEG",
            "filename": "shih_tzu.jpg",
        },
    ]

    print("Downloading test images...")
    print("=" * 60)

    downloaded = []
    for img in images:
        output_path = output_dir / img["filename"]
        if download_image(img["url"], output_path):
            downloaded.append(output_path)

    print("=" * 60)
    print(f"\nDownloaded {len(downloaded)}/{len(images)} images to {output_dir}")

    if downloaded:
        print("\nYou can now run:")
        print(
            f"uv run python examples/transparency_real_images.py {' '.join(str(p) for p in downloaded)}"  # noqa: E501
        )
    else:
        print("\nNo images were downloaded successfully.")


if __name__ == "__main__":
    main()
