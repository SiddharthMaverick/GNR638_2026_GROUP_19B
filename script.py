import argparse
import os
import pickle
import logging
import my_framework as nn
from model import LeNet
from import_time_script import load_dataset, get_batch, evaluate, train  # reuse functions from your existing file

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_model(weights_path=None):
    """Initialize LeNet and optionally load weights."""
    model = LeNet()
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        for p, saved_w in zip(model.parameters(), weights):
            for i in range(len(saved_w)):
                p.data[i] = saved_w[i]
        logger.info(f"[!] Loaded weights from {weights_path}")
    else:
        logger.info("[!] No weights loaded, starting fresh.")
    return model

def run_train(dataset_path, weights_path=None):
    logger.info("=== TRAINING MODE ===")
    model = load_model(weights_path)
    # reuse your train() function but allow dataset path override
    train(dataset_path=dataset_path, model=model)

def run_test(dataset_path, weights_path):
    logger.info("=== TESTING MODE ===")
    if not weights_path:
        logger.error("Testing requires a weights file path!")
        return
    model = load_model(weights_path)
    images, labels, _ = load_dataset(dataset_path)
    if not images:
        return
    acc = evaluate(model, images, labels, name="Test")
    logger.info(f"Final Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test LeNet model")
    parser.add_argument("--mode", choices=["train", "test"], required=True,
                        help="Choose whether to train or test the model")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset directory")
    parser.add_argument("--weights", default=None,
                        help="Optional path to model weights file")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.dataset, args.weights)
    elif args.mode == "test":
        run_test(args.dataset, args.weights)
