from datasets import load_dataset

# Load the caltech101 dataset from Hugging Face
dataset = load_dataset("dpdl-benchmark/dtd")

# Extract the class names from the training set features
class_labels = dataset["train"].features["label"].names

# Create a dictionary to map class label to class name
label_dict = {label: class_name for label, class_name in enumerate(class_labels)}

# Print the dictionary
print(label_dict)
