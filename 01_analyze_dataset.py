import os
import matplotlib.pyplot as plt

DATASET_DIR = "./dataset"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def analyze_dataset():
    class_counts = {}
    total_images = 0
    
    print("Dataset Distribution:")
    print("-" * 30)
    
    for c in CLASSES:
        path = os.path.join(DATASET_DIR, c)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            count = 0
        class_counts[c] = count
        total_images += count
        print(f"{c.capitalize():<12}: {count}")

    print("-" * 30)
    print(f"Total Images : {total_images}")

    # Plot using standard matplotlib without seaborn
    os.makedirs("./results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='#2b5c8f')
    plt.title("Garbage Classification Dataset Class Distribution")
    plt.xlabel("Waste Category")
    plt.ylabel("Image Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = "./results/class_distribution.png"
    plt.savefig(save_path)
    print(f"\nSaved class distribution plot to {save_path}")

if __name__ == "__main__":
    analyze_dataset()