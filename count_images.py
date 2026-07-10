import os

train_dir = "dermatology_dataset/train"
test_dir = "dermatology_dataset/test"

print("=== TRAINING SET ===")
total_train = 0
for cls in sorted(os.listdir(train_dir)):
    path = os.path.join(train_dir, cls)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        total_train += count
        print(f"  {cls}: {count}")
print(f"  TOTAL: {total_train}")

print("\n=== TEST SET ===")
total_test = 0
for cls in sorted(os.listdir(test_dir)):
    path = os.path.join(test_dir, cls)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        total_test += count
        print(f"  {cls}: {count}")
print(f"  TOTAL: {total_test}")
