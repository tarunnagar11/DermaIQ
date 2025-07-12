import os
import shutil

# ------------------- Paths -------------------
data_folder = r"E:\RECONNECT\data"
output_folder = r"E:\RECONNECT\classification_dataset"

# ------------------- Create output folders for each class -------------------
for label in ['0', '1', '2']:
    os.makedirs(os.path.join(output_folder, label), exist_ok=True)

# ------------------- Process each file -------------------
for file in os.listdir(data_folder):
    if file.lower().endswith('.jpg'):
        img_path = os.path.join(data_folder, file)
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                label_line = f.readline().strip()
            
            if label_line:  # Check if the line is not empty
                label_parts = label_line.split()
                
                if len(label_parts) > 0 and label_parts[0] in ['0', '1', '2']:
                    label = label_parts[0]
                    dest_folder = os.path.join(output_folder, label)
                    shutil.copy(img_path, dest_folder)
                else:
                    print(f"Warning: Label missing or incorrect format in {txt_path}")
            else:
                print(f"Warning: {txt_path} is empty")

print("Dataset classification folder created successfully ")
