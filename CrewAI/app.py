from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI  # Updated import
from langchain.tools import Tool  # Added for proper tool creation
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize configurations
TOPIC = "Medical Industry using generative AI"

# # Configure LLM using Groq
# llm = ChatOpenAI(
#     model="groq/Mixtral 8x7B",
#     temperature=0.7,
#     api_key=os.getenv("GROQ_API_KEY", "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"),
#     base_url="https://api.groq.com/openai/v1"
# )

# Initialize the SerperDev tool
search_tool = SerperDevTool(
    api_key=os.getenv("SERPER_API_KEY"),
    engine="google",
    num_results=10,
    search_type="news"
)

# Create agents
senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal="Conduct comprehensive research and analysis on generative AI applications in medicine",
    backstory="""You are an experienced medical research analyst specializing in AI applications.
    Your expertise includes analyzing clinical trials, research papers, and emerging technologies.
    You focus on evidence-based insights while maintaining ethical considerations.""",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[search_tool]  # Directly use the SerperDevTool instance
)

content_writer = Agent(
    role="Content Writer",
    goal="Create engaging, accurate medical content for diverse audiences",
    backstory="""You are a specialized medical content writer with expertise in AI technology.
    You excel at translating complex medical concepts into clear, engaging content.
    You maintain scientific accuracy while making content accessible to target audiences.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Create tasks with explicit agent assignments
research_task = Task(
    description=f"""
    Research and analyze the following topic: {TOPIC}
    
    Required steps:
    1. Search for recent applications of generative AI in medicine
    2. Analyze findings from medical journals and clinical trials
    3. Identify key trends and potential impacts
    4. Evaluate effectiveness and limitations
    5. Consider ethical implications and regulatory compliance
    """,
    agent=senior_research_analyst,
    expected_output="""
    A detailed report containing:
    - Comprehensive analysis of generative AI in medicine
    - Evidence-based insights from current research
    - Emerging trends and future predictions
    - Practical applications and limitations
    - References to key studies and sources
    """
)

content_creation_task = Task(
    description=f"""
    Using the research findings about {TOPIC}, create the following:
    
    1. Transform the research findings into engaging content
    2. Create differentiated content for medical professionals and public
    3. Develop educational materials with practical examples
    4. Include relevant case studies and success stories
    """,
    agent=content_writer,
    expected_output="""
    Deliverables:
    - Series of articles targeting different audience segments
    - Educational materials with practical applications
    - Clear explanations of complex concepts
    - Actionable insights for healthcare practitioners
    """
)

# Initialize and run crew
crew = Crew(
    agents=[senior_research_analyst, content_writer],
    tasks=[research_task, content_creation_task],
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