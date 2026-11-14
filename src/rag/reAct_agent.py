"""
ReAct agent setup for document retrieval and question answering.
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from src.llms.groq import llm
from src.rag.retriever_setup import get_retriever

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(REACT_PROMPT)


def get_agent_executor() -> AgentExecutor:
    """
    Initialize and return the ReAct agent executor with current tools.
    """
    tools = [get_retriever()]
    react_agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True,
        return_intermediate_steps=True
    )


class LazyAgentExecutor:
    """Lazy wrapper for backward compatibility with agent_executor imports."""
    def invoke(self, *args, **kwargs):
        executor = get_agent_executor()
        return executor.invoke(*args, **kwargs)


agent_executor = LazyAgentExecutor()
