from phi.agent import Agent
import os
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
from dotenv import load_dotenv
load_dotenv()


openai.api_key= os.getenv("OPENAI_API_KEY")
## Web search Agent

websearch_agent = Agent(
    name = "websearch AI Agent",
    role = "search the information from the web",
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview", api_key= os.getenv("GROQ_API_KEY")),
    tools = [DuckDuckGo()],
    instructions=["Alway include source of information in the search query"],
    show_tool_calls=True,
    markdown=True,
    )

## Financial Agent

financial_agent = Agent(
    name = "financial AI Agent",
    role="search the financial information",
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview", api_key= os.getenv("GROQ_API_KEY")),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Alway include source of information in the search query"],
    show_tool_calls=True,
    markdown=True, )


agent_team = Agent(
    team=[websearch_agent, financial_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)