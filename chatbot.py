# %%
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# %%
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Good morning", "Good evening", "Hey there"],
        "responses": ["Hello! How can I assist you today?", "Hi! What can I help you with?"],
    },
    {
        "tag": "repetitive_queries",
        "patterns": [
            "Why is my order delayed?",
            "How do I track my order?",
            "What are your working hours?",
            "What is your return policy?",
        ],
        "responses": [
            "You can track your order through our tracking system here: [tracking link].",
            "Our working hours are Monday to Friday, 9 AM to 5 PM.",
            "Our return policy is available at: [return policy link].",
        ],
    },
    {
        "tag": "language_barrier",
        "patterns": [
            "Do you support languages other than English?",
            "Can I get help in Spanish?",
            "Is there support for non-English queries?",
        ],
        "responses": [
            "Yes, we provide support in multiple languages. Please select your preferred language: [language options].",
            "We can assist you in Spanish and other languages. Please specify your query.",
        ],
    },
    {
        "tag": "resource_limitations",
        "patterns": [
            "Why does it take so long to get a response?",
            "Why is your customer support slow?",
            "Are you understaffed?",
        ],
        "responses": [
            "We apologize for any delays. We are working to improve response times by implementing automated systems.",
            "Our team is committed to resolving queries as quickly as possible. Thank you for your patience.",
        ],
    },
    {
        "tag": "customer_support_improvement",
        "patterns": [
            "What are you doing to improve customer support?",
            "How are you solving repetitive query problems?",
            "Are you using AI for customer support?",
        ],
        "responses": [
            "We are introducing AI-powered chatbots to handle repetitive queries, reducing response times and ensuring consistent support.",
            "Our team is implementing advanced tools to provide multilingual support and better manage high query volumes.",
        ],
    },
    {
        "tag": "farewell",
        "patterns": ["Bye", "Goodbye", "See you later", "Thanks, that's all"],
        "responses": ["Goodbye! Have a great day!", "Thanks for reaching out! Let us know if you need more help."],
    },
     {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Good morning", "Good evening", "Hey there"],
        "responses": ["Hello! How can I assist you today?", "Hi! What can I help you with?"],
    },
    {
        "tag": "order_status",
        "patterns": ["Where is my order?", "What is the status of my order?", "Track my order"],
        "responses": [
            "You can track your order here: [order tracking link].",
            "To check your order status, visit: [order status link]."
        ],
    },
    {
        "tag": "payment_issues",
        "patterns": [
            "Why was my payment declined?",
            "My payment failed, what do I do?",
            "What payment methods do you accept?"
        ],
        "responses": [
            "We accept credit cards, debit cards, PayPal, and other online payment methods. Details: [payment methods link].",
            "If your payment failed, please try again or contact your bank. Reach out to our support: [support link]."
        ],
    },
    {
        "tag": "farewell",
        "patterns": ["Bye", "Goodbye", "See you later", "Thanks, that's all"],
        "responses": ["Goodbye! Have a great day!", "Thanks for reaching out! Let us know if you need more help."],
    },
]


# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []

# Assuming 'intents' is a list of dictionaries with 'tag' and 'patterns' keys
for intent in intents:  
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns) 
y = tags
clf.fit(x, y)

print("Model trained successfully!")


# %%
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# %%
user_input = "Do you support languages other than English?"
response = chatbot(user_input)
print(response)

# %%
counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()


