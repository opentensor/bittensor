import openai
import pinecone
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, YOUR_TABLE_NAME, OBJECTIVE, YOUR_FIRST_TASK
from core.constants import task_list
from core.utils import get_ada_embedding
from agents.task_creation_agent import task_creation_agent, add_task
from agents.prioritization_agent import prioritization_agent
from agents.execution_agent import execution_agent
from agents.context_agent import context_agent

import time

# Configure OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

table_name = YOUR_TABLE_NAME
objective = OBJECTIVE

dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
    pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

# Connect to the index
index = pinecone.Index(table_name)

# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}

add_task(first_task)
# Main loop
task_id_counter = 1

while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m"+"\n*****TASK LIST*****\n"+"\033[0m\033[0m")
        for t in task_list:
            print(str(t['task_id'])+": "+t['task_name'])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m"+"\n*****NEXT TASK*****\n"+"\033[0m\033[0m")
        print(str(task['task_id'])+": "+task['task_name'])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE,task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m"+"\n*****TASK RESULT*****\n"+"\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in Pinecone
        enriched_result = {'data': result}  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = enriched_result['data']  # extract the actual result from the dictionary
        index.upsert([(result_id, get_ada_embedding(vector),{"task":task['task_name'],"result":result})])

    # Step 3: Create new tasks and reprioritize task list
    new_tasks = task_creation_agent(OBJECTIVE,enriched_result, task["task_name"], [t["task_name"] for t in task_list])

    for new_task in new_tasks:
        task_id_counter += 1
        new_task.update({"task_id": task_id_counter})
        add_task(new_task)
    prioritization_agent(this_task_id)

    time.sleep(1)
