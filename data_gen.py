import csv
from faker import Faker
import random

fake = Faker()

# Define fitness-related categories
categories = [1, 2, 3]

# Generate 200 fitness-related questions with categories
questions = []
for _ in range(200):
    if random.choice([True, False]):
        # Generate exercise-specific question
        question_text = f"What are some effective exercises for {fake.word()}?"
    else:
        # Generate general fitness question
        question_text = fake.sentence()
    category = random.choice(categories)
    questions.append([question_text, category])

# Write to CSV
with open('fitness_training_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Category'])
    writer.writerows(questions)
