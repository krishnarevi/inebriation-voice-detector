import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils import calculate_metrics, plot_results
from config import *  # Import all the config variables
from evaluate import evaluate

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    best_uar = 0
    no_improvement = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_uar': [], 'val_uar': []
    }
    
    for epoch in range(NUM_EPOCHS):  # Use NUM_EPOCHS from config
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Calculate training UAR using calculate_metrics
        train_metrics = calculate_metrics(train_labels, train_preds)
        train_uar = train_metrics['uar']
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion)
        
        # Update LR scheduler
        scheduler.step(val_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_uar'].append(train_uar)
        history['val_uar'].append(val_metrics['uar'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Train UAR: {train_uar:.4f} | Val UAR: {val_metrics['uar']:.4f}")
        print("Validation Confusion Matrix:")
        print(val_metrics['confusion_matrix'])
        
        # Early stopping check
        if val_metrics['uar'] > best_uar:
            best_uar = val_metrics['uar']
            no_improvement = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print("Saved new best model (based on UAR)!")
        else:
            no_improvement += 1
            print(f"No improvement in UAR for {no_improvement}/{PATIENCE} epochs")
            
            if no_improvement >= PATIENCE:
                print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement!")
                break
    
    # Load the best model weights before returning
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    print("\nTraining completed. Loaded best model weights.")
    
    return history
