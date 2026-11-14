"""
Graph builder module for the adaptive RAG system.
"""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.reAct_agent import agent_executor, get_agent_executor
from src.rag.retriever_setup import get_retriever
from src.config.settings import Config
from src.llms.groq import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool

config = Config()


# Node implementations
def query_classifier(state: State):
    """
    Classify the query to determine if it's related to indexed documents.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with route and latest_query.
    """
    question = state.get("latest_query") or state["messages"][-1].content
    retriever = get_retriever()
    try:
        context = retriever.invoke(question)
    except Exception as e:
        print(f"[query_classifier] Retriever invoke error: {e}")
        context = ""
    print("docs received from vector store:")
    print(context)

    route = "general"
    try:
        llm_with_structured_output = llm.with_structured_output(
            RouteIdentifier,
            method="function_calling"
        )    
        classify_prompt = PromptTemplate(
            template=config.prompt("classify_prompt"),
            input_variables=["question", "context"]
        )
        chain = classify_prompt | llm_with_structured_output
        result = chain.invoke({"question": question, "context": context})
        if result and hasattr(result, "route") and result.route:
            route = result.route
    except Exception as e:
        print(f"[query_classifier] Error in classification: {e}")
        if context and "No documents have been uploaded" not in context:
            route = "index"
        else:
            route = "general"

    print(f"[query_classifier] route: {route}")
    return {
        "messages": state["messages"],
        "route": route,
        "latest_query": question,
        "rewrite_count": state.get("rewrite_count") or 0,
    }


def general_llm(state: State):
    """
    Fetch general common knowledge result from the LLM.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages from LLM.
    """
    try:
        result = llm.invoke(state["messages"])
        print("inside general llm")
        print(result)
        return {"messages": result}
    except Exception as e:
        print(f"[general_llm] Error in general llm: {e}")
        return {"messages": [AIMessage(content="I apologize, but I encountered an error answering your question. Please try again.")]}


def retriever_node(state: State):
    """
    Retrieve results from vector stores using the reAct agent.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages with retrieved context and tool calls.
    """
    query = state.get("latest_query") or state["messages"][-1].content
    retriever_tool = get_retriever()
    context = ""
    tool_calls = []

    # Attempt retrieval using the ReAct agent executor
    try:
        executor = get_agent_executor()
        result = executor.invoke({"input": query})
        intermediate_steps = result.get("intermediate_steps", [])
        if intermediate_steps:
            for action, tool_result in intermediate_steps:
                tool_calls.append({
                    "tool": action.tool,
                    "input": action.tool_input,
                })
                context += f"\n{tool_result}"

        # If intermediate steps didn't produce context, check output
        if not context.strip() and result.get("output") and "Agent stopped" not in result["output"]:
            context = result["output"]
    except Exception as e:
        print(f"[retriever_node] ReAct agent error: {e}")

    # Fallback to direct retriever tool query if agent didn't retrieve context
    if not context.strip() or "Agent stopped" in context:
        try:
            print("[retriever_node] Fetching directly from retriever tool")
            context = retriever_tool.invoke(query)
            tool_calls.append({
                "tool": "retriever_customer_uploaded_documents",
                "input": query,
            })
        except Exception as e:
            print(f"[retriever_node] Direct retriever tool error: {e}")
            context = "No relevant document chunks found."

    new_message = AIMessage(
        content=context.strip(),
        additional_kwargs={"tool_calls": tool_calls},
    )

    return {
        "messages": [new_message]
    }


def grade(state: State):
    """
    Grade the results retrieved from vector stores.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with binary_score.
    """
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state.get("latest_query") or state["messages"][-1].content

    binary_score = "no"
    try:
        llm_with_grade = llm.with_structured_output(
            Grade,
            method="function_calling"
        )
        chain_graded = grading_prompt | llm_with_grade
        result = chain_graded.invoke({"question": question, "context": context})
        if result and hasattr(result, "binary_score") and result.binary_score:
            binary_score = result.binary_score.strip().lower()
            if "yes" in binary_score:
                binary_score = "yes"
            else:
                binary_score = "no"
    except Exception as e:
        print(f"[grade] Error during grading: {e}")
        # If context has meaningful retrieved text, treat as yes to avoid blocking answer
        if context and len(context) > 20 and "No relevant" not in context:
            binary_score = "yes"

    print(f"[grade] binary_score: {binary_score}")
    return {"messages": state["messages"], "binary_score": binary_score}


def rewrite_query(state: State):
    """
    Rewrite the query to get better retrieval results.

    Args:
        state (State): State of the question.

    Returns:
        dict: Updated latest_query and incremented rewrite_count.
    """
    query = state.get("latest_query") or state["messages"][-1].content
    rewrite_count = (state.get("rewrite_count") or 0) + 1
    try:
        rewrite_prompt = PromptTemplate(
            template=config.prompt("rewrite_prompt"),
            input_variables=["query"]
        )
        chain = rewrite_prompt | llm
        result = chain.invoke({"query": query})
        new_query = result.content.strip()
    except Exception as e:
        print(f"[rewrite_query] Error rewriting query: {e}")
        new_query = query

    print(f"[rewrite_query] Rewritten query: {new_query}, count: {rewrite_count}")
    return {
        "latest_query": new_query,
        "rewrite_count": rewrite_count,
    }


def generate(state: State):
    """
    Generate the final answer for the user.

    Args:
        state (State): State of the question.

    Returns:
        dict: Generated response.
    """
    context = state["messages"][-1].content
    try:
        generate_prompt = PromptTemplate(
            template=config.prompt("generate_prompt"),
            input_variables=["context"]
        )
        generate_chain = generate_prompt | llm
        result = generate_chain.invoke({"context": context})
        content = result.content
    except Exception as e:
        print(f"[generate] Error generating answer: {e}")
        content = context if context else "I was unable to find relevant information to answer your question."

    return {"messages": [{"role": "assistant", "content": content}]}


def web_search(state: State):
    """
    Search the web for the rewritten query.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Search results as messages.
    """
    query = state.get("latest_query") or state["messages"][-1].content
    contents = []
    try:
        search_tool = TavilySearchResults()
        result = search_tool.invoke(query)
        if isinstance(result, list):
            contents = [item["content"] for item in result if isinstance(item, dict) and "content" in item]
    except Exception as e:
        print(f"[web_search] Error searching web: {e}")

    content_str = "\n\n".join(contents) if contents else "No web search results found."
    return {
        "messages": [{"role": "assistant", "content": content_str}]
    }


# Build the graph
graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")
graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
graph.add_edge("generate", END)
graph.add_edge("general_llm", END)

builder = graph.compile()

