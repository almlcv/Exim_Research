from crewai import Agent, Task, Crew
from langchain_ollama.llms import OllamaLLM
import json
import os
from dotenv import load_dotenv

load_dotenv()

# # Load API keys from environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ollama_llm = OllamaLLM(model="llama3.2")

# # File path for JSON data
# JSON_FILE_PATH = "document_store/pdfs/np.json"

# ### AGENT 1: FILE READER ###
# file_reader_agent = Agent(
#     role="File Reader",
#     goal="Read JSON file content and pass it to the Data Extractor",
#     backstory="""You are a system agent that specializes in reading structured data from JSON files. 
#     Your role is to extract raw JSON content and provide it to the Data Extractor for further processing.""",
#     allow_delegation=True,
#     verbose=True,
#     llm=ollama_llm)



# ### AGENT 2: DATA EXTRACTOR ###
# data_extractor_agent = Agent(
#     role="Data Extractor",
#     goal= """Extract structured information of container number from JSON file using LLM processing, The information is given here = {
#                         "job_no": job["job_no"],
#                         "year": job["year"],
#                         "awb_bl_no": job["awb_bl_no"],
#                         "be_no": job["be_no"],
#                         "importer": job["importer"],
#                         "loading_port": job["loading_port"],
#                         "port_of_reporting": job["port_of_reporting"],
#                         "supplier_exporter": job["supplier_exporter"],
#                         "consignment_type": job["consignment_type"],
#                         "container_details": container ",}""", 


#     backstory="""You are a sophisticated AI agent that processes JSON data using advanced natural 
#     language models. Your job is to analyze and extract meaningful insights from JSON data based on 
#     user prompts""",
#     allow_delegation=False,
#     verbose=True,
#     llm=ollama_llm
# )

# ### TASK 1: READ JSON FILE ###
# def read_json_file():
#     try:
#         with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
#             json_data = json.load(file)
#         return json.dumps(json_data, indent=2)  # Convert JSON to string format
#     except Exception as e:
#         return f"Error reading JSON file: {str(e)}"

# file_reading_task = Task(
#     description=f"""
#     Read the JSON file located at {JSON_FILE_PATH} and return its content 
#     in structured format. Pass the JSON data to the Data Extractor for further processing.
#     """,
#     agent=file_reader_agent,
#     expected_output="Raw JSON data from the file"
# )

# data_extraction_task = Task(
#     description = f"""
#     Using the extracted JSON content, analyze and extract relevant insights based on a given prompt.
    
#     Steps:
#     1. Parse the JSON file data.
#     2. Identify key information based on structured prompts.
#     3. Use Ollama Llama3.2 for interpretation.
#     4. Provide meaningful insights based on the context.
    
#     Expected Output:
#     - Structured analysis of extracted data.
#     - Summarized insights based on AI processing.
#     """,
#     agent=data_extractor_agent,
#     expected_output="Structured analysis and extracted insights from JSON."
# )


# crew = Crew(
#     agents=[file_reader_agent, data_extractor_agent],
#     tasks=[file_reading_task, data_extraction_task],
#     verbose=True
# )

# def main():
#     try:
#         # Execute crew tasks
#         result = crew.kickoff()
#         print("\nCrew Execution Results:")
#         print(result)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         if hasattr(e, '__cause__') and e.__cause__:
#             print(f"Caused by: {str(e.__cause__)}")

# if __name__ == "__main__":
#     main()


import json
from crewai import Agent, Task, Crew
from crewai.llms import Ollama  # Corrected import for OllamaLLM

# Initialize Ollama LLM
ollama_llm = Ollama(model="llama2")  # Changed to llama2 as llama3.2 doesn't exist

# File path for JSON data
JSON_FILE_PATH = "document_store/pdfs/np.json"

### AGENT 1: FILE READER ###
Agent1 = Agent(
    role="File Reader",
    goal="Read JSON file content and pass it to the Data Extractor",
    backstory="""You are a system agent that specializes in reading structured data from JSON files. 
    Your role is to extract raw JSON content and provide it to the Data Extractor for further processing.""",
    allow_delegation=True,
    verbose=True,
    llm=ollama_llm
)

### AGENT 2: DATA EXTRACTOR ###
Agent2 = Agent(
    role="Data Extractor",
    goal="Extract structured information of container number from JSON file using LLM processing",
    tools=[
        {
            "key_fields": [
                "job_no",
                "year",
                "awb_bl_no",
                "be_no",
                "importer",
                "loading_port",
                "port_of_reporting",
                "supplier_exporter",
                "consignment_type",
                "container_details"
            ]
        }
    ],  # Added tools parameter to specify the fields to extract
    backstory="""You are a sophisticated AI agent that processes JSON data using advanced natural 
    language models. Your job is to analyze and extract meaningful insights from JSON data based on 
    user prompts""",
    allow_delegation=False,
    verbose=True,
    llm=ollama_llm
)

### TASK 1: READ JSON FILE ###
def read_json_file():
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        return json.dumps(json_data, indent=2)  # Convert JSON to string format
    except Exception as e:
        return f"Error reading JSON file: {str(e)}"

task1 = Task(
    description=f"""
    Read the JSON file located at {JSON_FILE_PATH} and return its content 
    in structured format. Pass the JSON data to the Data Extractor for further processing.
    """,
    agent=Agent1,
    context=read_json_file,  # Added context function
    expected_output="Raw JSON data from the file"
)

task2 = Task(
    description="""
    Using the extracted JSON content, analyze and extract relevant insights based on a given prompt.
    
    Steps:
    1. Parse the JSON file data.
    2. Identify key information based on structured prompts.
    3. Use Ollama LLM for interpretation.
    4. Provide meaningful insights based on the context.
    
    Expected Output:
    - Structured analysis of extracted data.
    - Summarized insights based on AI processing.
    """,
    agent=Agent2,
    context=lambda: task1.output,  # Added context from task1's output
    expected_output="Structured analysis and extracted insights from JSON."
)

# Create and configure the crew
crew = Crew(
    agents=[Agent1, Agent2],
    tasks=[task1, task2],
    verbose=True
)

def main():
    try:
        # Execute crew tasks
        result = crew.kickoff()
        print("\nCrew Execution Results:")
        print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Caused by: {str(e.__cause__)}")

if __name__ == "__main__":
    main()