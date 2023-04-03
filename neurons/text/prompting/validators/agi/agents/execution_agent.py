import openai
from agents.context_agent import context_agent
from core.config import YOUR_TABLE_NAME

def execution_agent(objective:str, task: str) -> str:
    context=context_agent(index=YOUR_TABLE_NAME, query=objective, n=5)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:",
        temperature=0.7,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()