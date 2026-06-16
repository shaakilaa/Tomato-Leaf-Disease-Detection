import os
import random
import shutil

# this code will move 10% of our train images to new test folder

# paths
dataset_path = "tomato-leaf-dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

test_size = 100

random.seed(44)

for class_name in os.listdir(train_path):

    train_class_path = os.path.join(train_path, class_name)
    test_class_path = os.path.join(test_path, class_name)

    if not os.path.isdir(train_class_path):
        continue

    os.makedirs(test_class_path, exist_ok=True)

    images = [
        f for f in os.listdir(train_class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_images = random.sample(images, test_size)

    for image in selected_images:
        src = os.path.join(train_class_path, image)
        dst = os.path.join(test_class_path, image)
        shutil.move(src, dst)

print("Test dataset created successfully!")