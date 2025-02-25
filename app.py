# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import AIMessage
from langchain.schema.runnable.config import RunnableConfig

load_dotenv()

# ChatOpenAI Templates
system_template = """You are a helpful assistant who will do the following:
1. Be clear and detailed
2. Stay relevant to the context of the question
3. You will perform Claim extraction to get multiple claims and statements from the news article that needs to be verified using Tavily, then perform Evidence search using Arxiv for the paper and publication that support or refute each of the claims and finally perform fact-checking using Google Search by matching all the claims with reliable sources such as government or fact checking websites that corroborate or debunk those claims.

Follow these guidelines while responding:
- Generate a report that includes all the claims made in different news articles
- Provide Evidence that was found to support or debunk each of the claim
- Finally an assessment of each claim based on research to be True, False, Partially True or Unverified

"""

user_template = """{input} 
"""

# 1. Initialize Tools
tavily_tool = TavilySearchResults(max_results=5)
google_search = Tool(
    name="GoogleSearch",
    func=GoogleSearchAPIWrapper().run, # Use the .run method directly
    description="Use this tool to search Google.", # Provide a description
)

tool_belt = [
    tavily_tool,
    ArxivQueryRun(),
    google_search,
]

# 2. Initialize and Bind the Model *BEFORE* starting the chat
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)

#Initialize state for LangGraph
class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

#add call_model and tool_node
def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  return {"messages" : [response]}

tool_node = ToolNode(tool_belt)

#add nodes to the graph
uncompiled_graph = StateGraph(AgentState)

uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)

uncompiled_graph.set_entry_point("agent")

#add conditional node
def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

uncompiled_graph.add_conditional_edges(
    "agent",
    should_continue
)

uncompiled_graph.add_edge("action", "agent")

# @cl.set_starters
# async def set_starters():
#     return [
#         cl.Starter(
#             label="Hi There! - Welcome, I am an AI Agent that can help with Fact-Checking News Articles",
#             message="Misinformation and Fake information are rampant online, Come here to get your facts right!!!! So, what do you want me to Fact-Check today?",
#             ),
#         ]

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    workflow = uncompiled_graph.compile()
    cl.user_session.set("workflow", workflow)

    greet_message = cl.Message(content="""**Hi There! - Welcome, I am an AI Agent that can help with Fact-Checking News Articles (Check out Readme to know more about me)**
    **Since, misinformation and fake information are rampant online, Come here to get your facts right!!!!**
    So, what do you want me to Fact-Check today?""")
    await greet_message.send()
    
@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    
    workflow = cl.user_session.get("workflow")
    config = {"configurable": {"thread_id": cl.context.session.id}}

    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    messages = [SystemMessage(system_template), HumanMessage(content=message.content)]

    for message, metadata in workflow.stream({
       "messages": messages
       }, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
       if (
           message.content
           and isinstance(message, AIMessage)
           and not message.tool_calls
       ):
           await final_answer.stream_token(message.content)


    await final_answer.send()
