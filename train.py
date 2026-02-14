import time
import os
import glob
import cv2
import math
import random 
import logging 
import pickle # <-- Added for saving model weights
import json   # <-- Added for saving training history
import my_framework as nn

from profiler import profile_model

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("training_metrics_data_1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 100
CLASSES = 10
IMG_SIZE = 32

def set_seed(seed=42):
    random.seed(seed)
    logger.info(f"Global seed set to {seed}")

# --- Checkpoint Utility ---
def save_checkpoint(model, filename):
    """Saves the model's parameters by converting ctypes to Python lists."""
    # Convert each parameter's C-array data into a standard Python list
    weights = [list(p.data) for p in model.parameters()]
    
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    logger.info(f"   [!] Model checkpoint saved to {filename}")

def load_checkpoint(model, filename):
    """Loads saved weights back into the model's ctypes parameters."""
    if not os.path.exists(filename):
        logger.info(f"   [!] Checkpoint {filename} not found.")
        return
        
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
        
    # Zip pairs the model's current parameters with the saved lists
    for p, saved_w in zip(model.parameters(), weights):
        for i in range(len(saved_w)):
            p.data[i] = saved_w[i] # Inject float back into the C-array
            
    logger.info(f"   [!] Model loaded successfully from {filename}")
    
    
# --- Metrics Utility ---
class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.confusion_matrix = [[0] * CLASSES for _ in range(CLASSES)]

    def update(self, logits, targets):
        batch_size = logits.shape[0]
        for i in range(batch_size):
            max_logit = -1e9
            pred_idx = -1
            for c in range(CLASSES):
                val = logits.data[i * CLASSES + c]
                if val > max_logit:
                    max_logit = val
                    pred_idx = c
            
            true_idx = -1
            for c in range(CLASSES):
                if targets.data[i * CLASSES + c] > 0.5:
                    true_idx = c
                    break
            
            if pred_idx == true_idx: self.correct += 1
            self.total += 1
            if 0 <= pred_idx < CLASSES and 0 <= true_idx < CLASSES:
                self.confusion_matrix[true_idx][pred_idx] += 1

    def print_report(self):
        accuracy = 100.0 * self.correct / self.total if self.total > 0 else 0
        logger.info(f"\nOverall Accuracy: {accuracy:.2f}% ({self.correct}/{self.total})")

        logger.info(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")

        logger.info("-" * 40)

       

        macro_p, macro_r, macro_f1 = 0, 0, 0

        for c in range(CLASSES):

            tp = self.confusion_matrix[c][c]

            fp = sum(self.confusion_matrix[x][c] for x in range(CLASSES)) - tp

            fn = sum(self.confusion_matrix[c][x] for x in range(CLASSES)) - tp

           

            p = tp / (tp + fp) if (tp + fp) > 0 else 0

            r = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

           

            logger.info(f"{c:<10} {p:.4f}     {r:.4f}     {f1:.4f}")

            macro_p += p; macro_r += r; macro_f1 += f1

           

        logger.info("-" * 40)

        logger.info(f"Macro Avg  {macro_p/CLASSES:.4f}     {macro_r/CLASSES:.4f}     {macro_f1/CLASSES:.4f}\n")

        
        return accuracy
    


# --- Dataset Loader ---
def load_dataset(root_path):
    logger.info(f"Loading dataset from {root_path}...")
    start_time = time.time()
    images, labels = [], []
    
    if not os.path.exists(root_path):
        logger.info(f"Error: Path '{root_path}' not found.")
        return [], [], 0

    class_folders = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
    
    if len(class_folders) == 0:
        logger.info("No class folders found! Check your directory structure.")
        return [], [], 0
        
    logger.info(f"Found {len(class_folders)} classes: {class_folders[:5]} ...") 
    
    total_files = 0
    
    for cls_name in class_folders:
        cls_path = os.path.join(root_path, cls_name)
        cls_idx = class_to_idx[cls_name]
        
        for f in glob.glob(os.path.join(cls_path, "*.png")):
            img = cv2.imread(f)
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            img_data = []
            for c in range(3): 
                for h in range(IMG_SIZE):
                    for w in range(IMG_SIZE):
                        img_data.append(float(img[h, w, c]) / 255.0)
            
            images.append(img_data)
            labels.append(cls_idx)
            total_files += 1

    loading_time = time.time() - start_time
    logger.info(f"Dataset loaded: {total_files} images in {loading_time:.2f} seconds.")
    return images, labels, loading_time

def get_batch(images, labels, idx, batch_size):
    end_idx = min(idx + batch_size, len(images))
    batch_imgs = images[idx : end_idx]
    batch_lbls = labels[idx : end_idx]
    
    flat_imgs = []
    for img in batch_imgs: flat_imgs.extend(img)
    
    curr_batch = len(batch_imgs)
    x = nn.Tensor((curr_batch, 3, IMG_SIZE, IMG_SIZE), data=flat_imgs)
    
    flat_targets = [0.0] * (curr_batch * CLASSES)
    for i, lbl in enumerate(batch_lbls):
        if 0 <= lbl < CLASSES: flat_targets[i * CLASSES + lbl] = 1.0
            
    y = nn.Tensor((curr_batch, CLASSES), data=flat_targets)
    return x, y

# --- Model Definition ---
class CNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=0) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, CLASSES)

    def forward(self, x):
        x = self.conv1(x); x = self.relu1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.relu2(x); x = self.pool2(x)
        x.shape = (x.shape[0], 16 * 5 * 5) 
        x = self.fc1(x); x = self.relu3(x)
        x = self.fc2(x); x = self.relu4(x)
        x = self.fc3(x)
        return x
        
    def parameters(self):
        return (self.conv1.parameters() + self.conv2.parameters() + 
                self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters())

# --- Evaluation ---
def evaluate(model, images, labels, name="Val"):
    logger.info(f"--- Evaluating on {name} ---")
    metrics = Metrics()
    
    for i in range(0, len(images), BATCH_SIZE):
        x_batch, y_batch = get_batch(images, labels, i, BATCH_SIZE)
        logits = model(x_batch) 
        metrics.update(logits, y_batch)
        
    return metrics.print_report() # Returns the accuracy



# --- Main ---
def train():
    full_imgs, full_lbls, _ = load_dataset("data_1")
    if not full_imgs: return

    combined = list(zip(full_imgs, full_lbls))
    random.shuffle(combined)
    full_imgs, full_lbls = zip(*combined)

    split_idx = int(len(full_imgs) * 0.8)
    train_imgs = list(full_imgs[:split_idx])
    train_lbls = list(full_lbls[:split_idx])
    val_imgs = list(full_imgs[split_idx:])
    val_lbls = list(full_lbls[split_idx:])

    model = CNN()
    optimizer = nn.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    profile_model(model, input_size=(3, 32, 32))
    logger.info(f"Training on {len(train_imgs)}, Validating on {len(val_imgs)}")

    # --- PROFESSIONAL TRACKING VARIABLES ---
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    best_val_acc = 0.0

    # ---> START MASTER TIMER HERE <---
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        metrics = Metrics()
        
        combined_train = list(zip(train_imgs, train_lbls))
        random.shuffle(combined_train)
        train_imgs, train_lbls = zip(*combined_train)
        
        num_batches = math.ceil(len(train_imgs) / BATCH_SIZE)
        
        for i in range(0, len(train_imgs), BATCH_SIZE):
            x_batch, y_batch = get_batch(train_imgs, train_lbls, i, BATCH_SIZE)
            logits = model(x_batch)
            loss, dlogits = criterion(logits, y_batch)
            metrics.update(logits, y_batch)
            
            optimizer.zero_grad()
            dout = model.fc3.backward(dlogits)
            dout = model.relu4.backward(dout)
            dout = model.fc2.backward(dout)
            dout = model.relu3.backward(dout)
            dout = model.fc1.backward(dout)
            dout.shape = (dout.shape[0], 16, 5, 5) 
            dout = model.pool2.backward(dout)
            dout = model.relu2.backward(dout)
            dout = model.conv2.backward(dout)
            dout = model.pool1.backward(dout)
            dout = model.relu1.backward(dout)
            model.conv1.backward(dout)
            optimizer.step()
            total_loss += loss
            
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        train_acc = 100 * metrics.correct / metrics.total
        
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS} | Time: {time.time()-start:.1f}s | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Evaluate and get validation accuracy
        val_acc = evaluate(model, val_imgs, val_lbls, f"Epoch {epoch+1} Val")
        
        # Update history
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, "best_model_data_1.pkl")

    # --- END OF TRAINING PRO STEPS ---
    
    # ---> STOP MASTER TIMER AND FORMAT IT <---
    total_time_seconds = time.time() - total_start_time
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    
    logger.info(f"\n=== Training Complete in {int(hours)}h {int(minutes)}m {seconds:.1f}s ===")
    
    # 1. Save Final Model
    save_checkpoint(model, "final_model_data_1.pkl")
    
    # 2. Save Training History
    # Also save the total time to your history json so you have a record of it!
    history["total_time_seconds"] = total_time_seconds 
    
    with open("training_history_data_1.json", "w") as f:
        json.dump(history, f, indent=4)
    logger.info("Saved training history to training_history_data_1.json")
    

if __name__ == "__main__":
    set_seed(42)
    train()