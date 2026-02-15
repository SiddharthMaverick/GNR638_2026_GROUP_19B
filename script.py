import argparse
import os
import pickle
import logging
import my_framework as nn
from model import LeNet
import train as train_module # Import as module so we can override its globals

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_model(num_classes, weights_path=None):
    """Initialize LeNet with dynamic classes and optionally load weights."""
    model = LeNet(num_classes=num_classes) # FIXED: Pass num_classes here
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
    if weights_path:
        logger.info(f"Pre-loaded checkpoint: {weights_path}")
    logger.info("Skipping training from script.py. Use train.py instead.")
    return

def run_test(dataset_path, weights_path, num_classes):
    logger.info("=== TESTING MODE ===")
    if not weights_path:
        logger.error("Testing requires a weights file path!")
        return
    model = load_model(num_classes, weights_path)
    
    # Load dataset
    full_images, full_labels, _ = train_module.load_dataset(dataset_path)
    if not full_images:
        logger.error("No images loaded from dataset.")
        return
    
    # Use SAME split as training (80/20) for consistency
    split_idx = int(len(full_images) * 0.8)
    test_images = list(full_images[split_idx:])  
    test_labels = list(full_labels[split_idx:])
    
    logger.info(f"Testing on {len(test_images)} images (validation split, same as training)")
    acc = train_module.evaluate(model, test_images, test_labels, CLASSES=num_classes, name="Test (Val Split)")
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
                        help="If set, test on ENTIRE dataset.")
    parser.add_argument("--classes", type=int, help="Number of classes")
    args = parser.parse_args()
    
    CLASSES = args.classes if args.classes else 10
    
    # CRITICAL FIX: Inject the class count into train.py's global scope 
    # so train_module.get_batch() formats the one-hot vectors correctly.
    train_module.CLASSES = CLASSES 
    
    if args.mode == "train":
        run_train(args.dataset, args.weights)
    elif args.mode == "test":
        if args.test_full:
            logger.info("=== TESTING MODE (FULL DATASET) ===")
            model = load_model(CLASSES, args.weights)
            images, labels, _ = train_module.load_dataset(args.dataset)
            if images:
                acc = train_module.evaluate(model, images, labels, CLASSES=CLASSES, name="Test (Full Dataset)")
                logger.info(f"Test Accuracy (Full): {acc:.2f}%")
        else:
            run_test(args.dataset, args.weights, CLASSES)