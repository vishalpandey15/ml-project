import pandas as pd
import random

# Load CSV mapping of UserB → UserA
# Make sure your CSV file has columns: userB,userA
df = pd.read_csv("userA_chats.csv")

# Function to generate reply
def generate_reply(userB_message):
    matches = df[df['userB'].str.lower() == userB_message.lower()]
    if not matches.empty:
        return matches.iloc[0]['userA']
    else:
        return random.choice(df['userA'].tolist())

# Test messages
test_messages = [
    "Hello! How are you?",
    "What are you doing?",
    "Python is great, right?",
    "Hi there!"
]

for msg in test_messages:
    reply = generate_reply(msg)
    print(f"UserB: {msg}")
    print(f"UserA: {reply}\n")
