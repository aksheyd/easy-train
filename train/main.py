import json
import tarfile
from train.config import CHECKPOINT_PATH, TRAINING_COMPLETED
import train.sl_loop
import chz
from dotenv import load_dotenv
import sys
import os
import tinker
import urllib.request


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

if __name__ == "__main__":
    if not TRAINING_COMPLETED:
        checkpoint_dict = chz.nested_entrypoint(train.sl_loop.main)
        TRAINING_COMPLETED = True
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint_dict, f)
    else:
        with open(CHECKPOINT_PATH, "r") as f:
            checkpoint_dict = json.load(f)

    sampler_weights_path = checkpoint_dict["sampler_path"]
    if not os.path.exists("sampler_weights.tar"):
        print("Sampler weights not found, downloading...")
        sc = tinker.ServiceClient()
        rc = sc.create_rest_client()
        future = rc.get_checkpoint_archive_url_from_tinker_path(sampler_weights_path)
        checkpoint_archive_url_response = future.result()
        urllib.request.urlretrieve(
            checkpoint_archive_url_response.url, "sampler_weights.tar"
        )
        print("Sampler weights downloaded")

    if not os.path.exists("sampler_weights"):
        os.makedirs("sampler_weights")
        tar = tarfile.open("sampler_weights.tar")
        tar.extractall("sampler_weights")
        tar.close()

    print("Now upload the sampler weights to Hugging Face Hub")
