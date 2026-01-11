import operator
from typing import Sequence, Annotated
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import BaseMessage


class ResearchInputState(MessagesState):
    pass


class ResearchState(MessagesState):
    research_brief: str | None
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add]
    final_report: str
