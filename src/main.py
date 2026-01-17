from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

import asyncio
import state
import nodes


async def main():
    graph = StateGraph(state_schema=state.ResearchInputState)

    graph = graph.add_node(nodes.clarify_user_request)
    graph = graph.add_node(nodes.write_research_brief)
    graph = graph.add_edge("__start__", nodes.clarify_user_request.__name__)
    compiled_graph = graph.compile(checkpointer=InMemorySaver())

    return compiled_graph


if __name__ == "__main__":
    asyncio.run(main())
