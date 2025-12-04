#!/usr/bin/env python3
"""
Machine Unlearning Demo Script
Converted from demo.ipynb
"""

import os
import torch
import torchunlearn
from torchunlearn.utils.data import UnlearnDataSetup, MergedLoaders
from torchunlearn.unlearn.trainers.finetune import Finetune

def main():
    print("=== Machine Unlearning Demo ===")
    
    # Set up environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Possible GPUs
    
    # Configuration
    PATH = "./models/"
    NAME = "Test"
    SAVE_PATH = PATH + NAME
    
    MODEL_NAME = "ResNet18"
    DATA_NAME = "CIFAR10"
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    BATCH_SIZE = 128
    N_CLASSES = 10
    N_VALIDATION = 1000
    EPOCH = 5
    
    PRETRAINED_PATH = PATH + f"{DATA_NAME}_Standard/last.pth"
    
    print(f"Package version: {torchunlearn.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n=== Prepare Datasets ===")
    
    # Setup data
    setup = UnlearnDataSetup(data_name=DATA_NAME, n_classes=N_CLASSES, mean=MEAN, std=STD)
    
    print("\n--- For Random-forgetting ---")
    train_loaders, test_loaders = setup.get_loaders_for_rand(
        batch_size=BATCH_SIZE, 
        ratio=0.1, 
        stratified=True,
        train_shuffle_and_transform=False, 
        drop_last_train=True, 
        seed=42
    )
    
    print("\n--- For Classwise-forgetting ---")
    train_loaders_classwise, test_loaders_classwise = setup.get_loaders_for_classwise(
        batch_size=BATCH_SIZE, 
        omit_label=1,
        train_shuffle_and_transform=True, 
        drop_last_train=True
    )
    
    print("\n=== Prepare Model ===")
    
    # Load model - using CPU for RTX 5080 compatibility
    device = torch.device('cpu')  # Force CPU due to RTX 5080 compatibility issues
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("Note: Using CPU due to RTX 5080 architecture compatibility issues")
    
    model = torchunlearn.utils.load_model(model_name=MODEL_NAME, n_classes=N_CLASSES).to(device)
    rmodel = torchunlearn.RobModel(
        model, 
        n_classes=N_CLASSES, 
        normalization_used={'mean': MEAN, 'std': STD}
    ).to(device)
    
    # Load pretrained weights if available
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading pretrained model from: {PRETRAINED_PATH}")
        try:
            # Use weights_only=False for compatibility with older checkpoints
            state_dict = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
            rmodel.load_state_dict_auto(state_dict["rmodel"])
            print("Model loaded successfully.")
            if "record_info" in state_dict.keys():
                print("Record Info:")
                print(state_dict["record_info"])
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Continuing with randomly initialized model...")
    else:
        print(f"Pretrained model not found at: {PRETRAINED_PATH}")
        print("Continuing with randomly initialized model...")
    
    print("\n=== Start Unlearn ===")
    
    # Setup trainer
    trainer = Finetune(rmodel)
    
    # Create merged loader
    merged_train_loader = MergedLoaders(train_loaders)
    
    # Setup training configuration
    trainer.setup(
        optimizer=f"SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)",
        scheduler=None, 
        scheduler_type=None,
        minimizer=None, 
        n_epochs=EPOCH,
    )
    
    # Setup evaluation loaders
    loaders_with_flags = {
        "(R)": train_loaders['Retain'],
        "(F)": train_loaders['Forget'],
        "(Te)": test_loaders['Test'],
    }
    
    trainer.record_rob(loaders_with_flags, n_limit=N_VALIDATION)
    
    # Start training
    print("\nStarting unlearning process...")
    trainer.fit(
        train_loaders=merged_train_loader, 
        n_epochs=EPOCH,
        save_path=SAVE_PATH, 
        save_best={"Clean(R)": "HB", "Clean(F)": "LBO"},
        save_type=None, 
        save_overwrite=True, 
        record_type="Epoch"
    )
    
    print("\n=== Demo Complete ===")
    print("Results saved to:", SAVE_PATH)

if __name__ == "__main__":
    main()
