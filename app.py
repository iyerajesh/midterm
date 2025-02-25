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
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document



load_dotenv()

# RAG implementation begins here

path = "data/"
loader = DirectoryLoader(path, glob="*.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(docs)
len(split_documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="obecity_rag",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="obecity_rag",
    embedding=embeddings,
)

_ = vector_store.add_documents(documents=split_documents)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def retrieve(state):
  retrieved_docs = retriever.invoke(state["question"])
  return {"context" : retrieved_docs}

RAG_PROMPT = """\
You are a helpful assistant who will do the following:
1. Be clear and detailed
2. Stay relevant to the context of the question

Follow these guidelines while responding:
- Assist in setting realistic and achievable weight-loss goals that are tailored to individual [needs] and [lifestyle]. The process should involve an initial assessment of current habits, health status, and lifestyle to establish a baseline. From there, develop a structured, step-by-step plan that includes short-term milestones and long-term objectives. The plan should be flexible enough to adjust as progress is made but structured enough to provide clear direction. Incorporate strategies for overcoming common obstacles, such as motivation dips and plateaus, and recommend tools or resources for tracking progress. Ensure the goals are SMART (Specific, Measurable, Achievable, Relevant, and Time-bound) to increase the likelihood of success.
- Your task is to identify and help address unhelpful eating patterns in the client seeking to improve their health and wellness. Begin by conducting a comprehensive assessment to understand the client's current eating habits, lifestyle, and underlying factors contributing to their eating patterns. Develop a personalized plan that incorporates achievable goals, mindful eating strategies, and healthier food choices. Provide ongoing support, motivation, and adjustments to the plan based on the client’s progress and feedback. Your approach should be empathetic, evidence-based, and tailored to each client's unique needs, aiming to foster sustainable, positive changes in their eating habits.
- Act as a fitness coach. Develop a personalized workout routine specifically tailored to meet the client's [fitness goal]. The routine must consider the client's current fitness level, any potential limitations or injuries, and their available equipment. It should include a mix of cardiovascular exercises, strength training, flexibility workouts, and recovery activities. Provide clear instructions for each exercise, suggest the number of sets and repetitions, and offer guidance on proper form to maximize effectiveness and minimize the risk of injury.
- As a Personal Chef specialized in creating customized meal plans, design a meal plan tailored to specific dietary preferences. This plan should cater to the client's [health goals], [taste preferences], and any [dietary restrictions] they might have. The meal plan should cover breakfast, lunch, dinner, and snack options for one week, ensuring a balanced and nutritious diet. Include a detailed list of ingredients for each meal, preparation instructions that are easy to follow, and tips for meal prepping to save time.

### Question
{question}

### Context
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
llm = ChatOpenAI(model="gpt-4o-mini")

def generate(state):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = llm.invoke(messages)
  return {"response" : response.content}

class State(TypedDict):
  question: str
  context: List[Document]
  response: str
  
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
  
response = graph.invoke({"question" : "Why is obesity a big problem in America?"})
response["response"]

#@Tool (name="Obesity Question Answering Tool", description="Useful for when you need to answer questions about obesity. Input should be a fully formed question.")
def ai_rag_tool(question: str) -> str:
    """Useful for when you need to answer questions about obesity. Input should be a fully formed question."""
    response = graph.invoke({"question" : question})
    return {
        "messages" : [HumanMessage(content=response["response"])],
        "context" : response["context"]
    }

ai_rag_tool_instance = Tool(
    name="Obesity_QA_Tool",  # ✅ No spaces, only letters, numbers, underscores, or hyphens
    description="Useful for when you need to answer questions about obesity. Input should be a fully formed question.",
    func=ai_rag_tool
)

# RAG implementation ends here  

# Search implementation begins here

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
    ai_rag_tool_instance,
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
