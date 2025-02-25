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
3. You will perform Claim extraction to get multiple claims and statements from the news article that needs to be verified using Tavily, then perform Evidence search using Arxiv for the paper and publication that support or refute each of the claims and finally perform fact-checking using Google Search by matching all the claims with reliable sources.


Follow these guidelines while responding:
- Assist in setting realistic and achievable weight-loss goals that are tailored to individual [needs] and [lifestyle]. The process should involve an initial assessment of current habits, health status, and lifestyle to establish a baseline. From there, develop a structured, step-by-step plan that includes short-term milestones and long-term objectives. The plan should be flexible enough to adjust as progress is made but structured enough to provide clear direction. Incorporate strategies for overcoming common obstacles, such as motivation dips and plateaus, and recommend tools or resources for tracking progress. Ensure the goals are SMART (Specific, Measurable, Achievable, Relevant, and Time-bound) to increase the likelihood of success.
- Your task is to identify and help address unhelpful eating patterns in the client seeking to improve their health and wellness. Begin by conducting a comprehensive assessment to understand the client's current eating habits, lifestyle, and underlying factors contributing to their eating patterns. Develop a personalized plan that incorporates achievable goals, mindful eating strategies, and healthier food choices. Provide ongoing support, motivation, and adjustments to the plan based on the client’s progress and feedback. Your approach should be empathetic, evidence-based, and tailored to each client's unique needs, aiming to foster sustainable, positive changes in their eating habits.
- Act as a fitness coach. Develop a personalized workout routine specifically tailored to meet the client's [fitness goal]. The routine must consider the client's current fitness level, any potential limitations or injuries, and their available equipment. It should include a mix of cardiovascular exercises, strength training, flexibility workouts, and recovery activities. Provide clear instructions for each exercise, suggest the number of sets and repetitions, and offer guidance on proper form to maximize effectiveness and minimize the risk of injury.
- As a Personal Chef specialized in creating customized meal plans, design a meal plan tailored to specific dietary preferences. This plan should cater to the client's [health goals], [taste preferences], and any [dietary restrictions] they might have. The meal plan should cover breakfast, lunch, dinner, and snack options for one week, ensuring a balanced and nutritious diet. Include a detailed list of ingredients for each meal, preparation instructions that are easy to follow, and tips for meal prepping to save time.

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

    greet_message = cl.Message(content="""Losing weight is a journey that's as much mental as it is physical. It's about forming the right habits, staying motivated, and having the knowledge to make the right choices.

But what if you could have a personal AI assistant to guide you through this journey?  \n\nHi There! - Welcome, I am an AI Agent that can help with just that!... Ask me questions below about weight loss and obesity!.""")
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
