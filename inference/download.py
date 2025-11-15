"""Download checkpoint from Tinker."""

import tinker
import urllib.request
import tarfile
from pathlib import Path


def download_checkpoint(checkpoint_path: str, output_dir: str = "downloaded_weights"):
    """
    Download checkpoint from Tinker.

    Educational note:
    ----------------
    Tinker stores checkpoints as signed URLs that expire.
    This downloads the LoRA adapter in PEFT format:
    - adapter_config.json: LoRA configuration
    - adapter_model.safetensors: Adapter weights

    Args:
        checkpoint_path: Tinker checkpoint path (tinker://{uuid}/sampler_weights/final)
        output_dir: Local directory to save checkpoint

    Returns:
        Path to downloaded checkpoint directory
    """
    print(f"\n📥 Downloading checkpoint: {checkpoint_path}")

    # Create service client
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    # Get signed URL
    future = rc.get_checkpoint_archive_url_from_tinker_path(checkpoint_path)
    url_response = future.result()

    print(f"  URL expires: {url_response.expires}")

    # Download archive
    archive_path = "checkpoint.tar"
    print(f"  Downloading...")
    urllib.request.urlretrieve(url_response.url, archive_path)

    # Extract
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting to {output_dir}...")
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(output_dir)

    # Cleanup
    Path(archive_path).unlink()

    print(f"✅ Downloaded to {output_dir}")
    print(f"  Files: {list(output_path.iterdir())}")

    return str(output_path)
