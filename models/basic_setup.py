from datasets import load_dataset

food = load_dataset("food101", split="train[:5000]")
food = food.train_test_split(test_size=0.2)
food["train"][0]




from transformers import pipeline

classifier = pipeline(task="image-classification")
classifier()

