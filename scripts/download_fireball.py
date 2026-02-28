from datasets import load_dataset

# Load the dataset (you may need to pass trust_remote_code=True because of the custom script)
dataset = load_dataset("lara-martin/FIREBALL", trust_remote_code=True)

# Take a look at the structure
print(dataset)