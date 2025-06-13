from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
import transformers
import torch
from langchain.prompts.base import StringPromptTemplate
from typing import List


class ConversationManager:
    def __init__(self, model, tokenizer):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                
        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=1024,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )


        llm = HuggingFacePipeline(pipeline=pipe)

        class SimpleChatFormatter(StringPromptTemplate):
            def format(self, **kwargs) -> str:
                chat_history: List = kwargs["chat_history"]
                input_text: str = kwargs["input"]

                formatted_history = ""
                for msg in chat_history:
                    role = "User" if msg.type == "human" else "Assistant"
                    formatted_history += f"{role}: {msg.content}\n"

                return f"{formatted_history}User: {input_text}\nAssistant:"

        prompt = SimpleChatFormatter(input_variables=["chat_history", "input"])



        self.chain = LLMChain(llm=llm, prompt=prompt, memory=self.memory)

    def get_response(self, user_input):
        return self.chain.predict(input=user_input)

    def clear(self):
        self.memory.clear()
