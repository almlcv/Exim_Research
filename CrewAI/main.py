from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize configurations
TOPIC = "Medical Industry using generative AI"

# Configure LLM using GPT-4
llm = ChatOpenAI(
    model="gpt-4",  # Using GPT-4 model
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")  # Make sure to use OpenAI API key
)

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
    tools=[search_tool]
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
    
    Focus on finding recent, concrete examples of generative AI applications
    in healthcare settings and their measured outcomes.
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
    
    Ensure to maintain scientific accuracy while making the content 
    accessible to different audience levels.
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