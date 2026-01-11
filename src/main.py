from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.graph.state import StateGraph
from utils import get_readable_date

import asyncio
import prompts
import state
import nodes


async def main() -> None:
    graph = StateGraph(state_schema=state.ResearchInputState)

    input = state.ResearchInputState(
        messages=[HumanMessage(content="What are the best coffee shops in Warsaw?")]
    )
    graph = graph.add_node(nodes.clarify_user_request)
    graph = graph.add_node(nodes.write_research_brief)
    graph = graph.add_edge("__start__", nodes.clarify_user_request.__name__)
    compiled_graph = graph.compile()

    result = await compiled_graph.ainvoke(input)

    print(get_buffer_string(result["messages"]))


if __name__ == "__main__":
    asyncio.run(main())
