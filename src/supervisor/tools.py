from ..ctx import Context

# from langgraph.runtime import Runtime
from langchain.tools import ToolRuntime, tool


@tool
async def invoke_researcher(research_topic: str, runtime: ToolRuntime[Context]) -> str:
    """
    Call this tool to conduct research on a specific topic.

    :param research_topic: The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).
    :type research_topic: str
    """

    response = await runtime.context.researcher.ainvoke(
        {"research_topic": research_topic}
    )

    return response.get(
        "compressed_research", "The researcher failed to complete its job"
    )
