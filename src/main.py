from functools import partial
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
import dotenv

import asyncio
from supervisor import supervisor_state as supervisor_state
from researcher import researcher_state as researcher_state
import supervisor.supervisor as supervisor
import researcher.researcher as researcher


async def main():
    dotenv.load_dotenv()

    checkpointer = InMemorySaver()

    researcher_graph = StateGraph(state_schema=researcher_state.ResearcherState)
    researcher_graph.add_node(researcher.research)
    researcher_graph.add_node(researcher.handle_researcher_tools)
    researcher_graph.add_node(researcher.compress_research)
    researcher_graph.add_edge("__start__", researcher.research.__name__)

    researcher_graph = researcher_graph.compile(checkpointer=checkpointer)

    supervisor_graph = StateGraph(state_schema=supervisor_state.SillySearchState)

    supervisor_graph.add_node(supervisor.clarify_user_request)
    supervisor_graph.add_node(supervisor.write_research_brief)
    supervisor_graph.add_node(supervisor.supervise)
    supervisor_graph.add_node(
        supervisor.handle_supervisor_tools.__name__,
        partial(
            supervisor.handle_supervisor_tools,
            researcher=researcher_graph,
        ),
    )
    supervisor_graph.add_edge("__start__", supervisor.clarify_user_request.__name__)
    supervisor_graph = supervisor_graph.compile(checkpointer=checkpointer)

    return supervisor_graph


if __name__ == "__main__":
    asyncio.run(main())
