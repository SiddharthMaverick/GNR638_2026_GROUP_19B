import argparse
import os
import pickle
import logging
import my_framework as nn
from model import LeNet
from train import load_dataset, get_batch, evaluate, train  # reuse functions from your existing file

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
    logger.info(f"Training on dataset: {dataset_path}")
    logger.info(f"Using weights from: {weights_path if weights_path else 'scratch'}")
    logger.info("Note: Use train.py directly with CLI args for full control over hyperparameters")
    logger.info("Example: python train.py --data-path data_1 --epochs 10 --batch-size 32")
    # For now, log that user should use train.py directly
    if weights_path:
        logger.info(f"Pre-loaded checkpoint: {weights_path}")
    logger.info("Skipping training from script.py. Use train.py instead.")
    return

def run_test(dataset_path, weights_path):
    logger.info("=== TESTING MODE ===")
    if not weights_path:
        logger.error("Testing requires a weights file path!")
        return
    model = load_model(weights_path)
    
    # Load dataset
    full_images, full_labels, _ = load_dataset(dataset_path)
    if not full_images:
        logger.error("No images loaded from dataset.")
        return
    
    # Use SAME split as training (80/20) for consistency
    # Testing on validation split (last 20%) like training did
    split_idx = int(len(full_images) * 0.8)
    test_images = list(full_images[split_idx:])  # Use validation split for testing
    test_labels = list(full_labels[split_idx:])
    
    logger.info(f"Testing on {len(test_images)} images (validation split, same as training)")
    acc = evaluate(model, test_images, test_labels, name="Test (Val Split)")
    logger.info(f"Final Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test LeNet model")
    parser.add_argument("--mode", choices=["train", "test"], required=True,
                        help="Choose whether to train or test the model")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset directory")
    parser.add_argument("--weights", default=None,
                        help="Optional path to model weights file")
    parser.add_argument("--test-full", action='store_true', 
                        help="If set, test on ENTIRE dataset. Otherwise tests on validation split (consistent with training).")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.dataset, args.weights)
    elif args.mode == "test":
        if args.test_full:
            # Alternative: test on full dataset
            logger.info("=== TESTING MODE (FULL DATASET) ===")
            model = load_model(args.weights)
            images, labels, _ = load_dataset(args.dataset)
            if images:
                acc = evaluate(model, images, labels, name="Test (Full Dataset)")
                logger.info(f"Test Accuracy (Full): {acc:.2f}%")
        else:
            run_test(args.dataset, args.weights)
