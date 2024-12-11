from knowledge_base import colleges
from chatbot import CollegeChatbot

def main():
    # Initialize the chatbot
    chatbot = CollegeChatbot(colleges)
    
    print("College Information Chatbot")
    print("Type 'exit' to end the conversation")
    
    while True:
        # Get user input
        user_query = input("\nYou: ")
        
        # Check for exit condition
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Generate and print response
        response = chatbot.generate_response(user_query)
        print("\nChatbot:", response)

if __name__ == "__main__":
    main()