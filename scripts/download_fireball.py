from datasets import load_dataset

# Load the dataset
dataset = load_dataset("lara-martin/FIREBALL", trust_remote_code=True)

# Take a look at the structure
print(dataset)
print("Test")