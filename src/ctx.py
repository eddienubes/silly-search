from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from pydantic import typing
from researcher.state import ResearcherState


@dataclass
class Context:
    researcher: CompiledStateGraph[typing.Any, typing.Any, typing.Any, ResearcherState]
