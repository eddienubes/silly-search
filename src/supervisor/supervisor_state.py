import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState, add_messages


class SillySearchState(MessagesState):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_brief: str
    raw_notes: Annotated[list[str], operator.add]
    notes: Annotated[list[str], operator.add]
    final_report: str


class SupervisorState(TypedDict):
    research_brief: str | None
    supervisor_messages: Annotated[Sequence[MessageLikeRepresentation], add_messages]
    notes: Annotated[list[str], operator.add]
    raw_notes: Annotated[list[str], operator.add]
    final_report: str
    research_iterations: int


