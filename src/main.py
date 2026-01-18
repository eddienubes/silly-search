from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
import dotenv

import asyncio
from supervisor import state as supervisor_state
from researcher import state as researcher_state
import supervisor.superviser as superviser
import researcher.researcher as researcher


async def main():
    dotenv.load_dotenv()

    researcher_graph = StateGraph(state_schema=researcher_state.ResearcherState)
    researcher_graph.add_node(researcher.research)
    researcher_graph.add_node(researcher.handle_researcher_tools)
    researcher_graph.add_node(researcher.compress_research)
    researcher_graph.add_edge("__start__", researcher.research.__name__)

    supervisor_graph = StateGraph(state_schema=supervisor_state.SillySearchInput)

    supervisor_graph = supervisor_graph.add_node(superviser.clarify_user_request)
    supervisor_graph = supervisor_graph.add_node(superviser.write_research_brief)
    supervisor_graph = supervisor_graph.add_node(superviser.supervise)
    supervisor_graph = supervisor_graph.add_edge(
        "__start__", superviser.clarify_user_request.__name__
    )
    compiled_supervisor_graph = supervisor_graph.compile(checkpointer=InMemorySaver())

    return compiled_supervisor_graph


if __name__ == "__main__":
    asyncio.run(main())
