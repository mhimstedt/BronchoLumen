import os


folder_path = "labels"  
# Process each .txt file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        new_lines = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id in [8, 9]:
                    continue  # Remove lines with class 8 or 9
                # Replace all other class IDs with 0
                new_line = "0 " + " ".join(parts[1:])
                new_lines.append(new_line)

        # Overwrite the original file with cleaned content
        with open(file_path, 'w') as f:
            for line in new_lines:
                f.write(line + "\n")

print("All files processed successfully.")
