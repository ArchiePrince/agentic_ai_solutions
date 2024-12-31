# Title of Project: Financial AI Agent
# Author: Archie Prince
# Filename: finance_agent.py
# Date: 31.12.2024
# Objective: To create a seamless financial AI agent workflow to predict and make financial decisions as necessary

## Call Agent, Models and Tools needed for the workflow
import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()
Groq.api_key = os.getenv("GROQ_API_KEY")


## Duckduckgo Web Search Agent

web_searcher = Agent(
    name = "Duckduckgo Web Search Agent",
    role = "To search the web for information",
    model = Groq(id = "llama3-groq-8b-8192-tool-use-preview"),
    tools = [DuckDuckGo()],
    instructions=["Remember to always include sources"],
    show_tool_calls=True,
    markdown=True
    )



## YFinance Tool Agent
finance_researcher = Agent(
    name="Financial AI Agent Analyst",
    role="Gets top information about a topic",
    tools=[
        YFinanceTools(company_news=True, stock_price=True, analyst_recommendations=True, stock_fundamentals=True)
        ],
    model = Groq(id = "llama3-groq-8b-8192-tool-use-preview"),
    instructions=["Format your response using markdown and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True,

)


## This assembles the two agents as a team lead

finance_team = Agent(
    name="Financial Analysis Team",
    team=[web_searcher, finance_researcher],
    model = Groq(id = "llama-3.1-70b-versatile"),
    instructions=["Remember to always include sources", "Format your response using markdown and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True,
)

finance_team.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)

