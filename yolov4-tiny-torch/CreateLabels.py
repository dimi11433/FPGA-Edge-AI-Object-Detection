import os

input_file = "dataset/images/val/_annotations.txt"
output_dir = "dataset/labels/val"

os.makedirs(output_dir, exist_ok=True)


with open(input_file, "r") as file_object:
    for line in file_object:
        parts = line.strip().split(" ")
        if len(parts) < 2:
            continue
        image_name = parts[0]
        farts = parts[1].split(",")
        coords = " ".join(farts)
        
        # create a label file with same name but .txt
        label_filename = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(output_dir, label_filename)

        with open(label_path, "a") as f:  # "a" in case multiple objects per image
            f.write(coords + "\n")
            
    
    



            
            
        
