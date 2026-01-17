import operator
from typing import Annotated, TypedDict

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import add_messages


class ResearcherState(TypedDict):
    researcher_messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_brief: str
    notes: Annotated[list[str], operator.add]
    research_iterations: int
    raw_notes: Annotated[list[str], operator.add]
