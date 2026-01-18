# from langgraph.runtime import Runtime
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from langgraph.graph.state import CompiledStateGraph
import typing


@tool
async def invoke_researcher(
    research_topic: str,
    researcher: typing.Annotated[CompiledStateGraph, InjectedToolArg],
) -> str:
    """
    Call this tool to conduct research on a specific topic.

    :param research_topic: The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).
    :type research_topic: str
    """

    response = await researcher.ainvoke(input={"research_topic": research_topic})

    return response.get(
        "compressed_research", "The researcher failed to complete its job"
    )
