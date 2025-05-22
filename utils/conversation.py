import os
from langchain.memory import ConversationBufferMemory

class ConversationManager:
    """
    Manages conversation history and context for the chatbot.
    Provides session management and context retrieval for multi-turn dialogues.
    """
    
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def get_context(self, conversation_history):
        """
        Extract relevant context from conversation history
        to provide to the IBM Granite model
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            String containing relevant context
        """
        # For simplicity, we'll include the last 3 exchanges
        relevant_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        # Format context for the model
        context_text = ""
        for message in relevant_history:
            role = "Human" if message["role"] == "user" else "Assistant"
            context_text += f"{role}: {message['content']}\n"
            
        return context_text.strip()
    
    def update_context(self, conversation_history):
        """
        Update the conversation memory with the latest exchanges
        
        Args:
            conversation_history: List of conversation messages
        """
        # Extract the latest exchange if available
        if len(conversation_history) >= 2:
            latest_user = conversation_history[-2]["content"]
            latest_assistant = conversation_history[-1]["content"]
            
            # Update memory
            self.memory.save_context({"input": latest_user}, {"output": latest_assistant})
    
    def get_summary(self):
        """
        Get a summary of the conversation for routing or classification
        
        Returns:
            String summary of key conversation points
        """
        # In a real implementation, you might use the IBM Granite model to generate
        # a summary of the conversation here
        return self.memory.load_memory_variables({}).get("history", "")
        
    def clear(self):
        """
        Clear the conversation memory
        """
        self.memory.clear()