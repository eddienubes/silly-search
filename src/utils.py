from datetime import datetime
from collections.abc import Callable
import typing

from langchain_core.runnables import RunnableConfig


def get_readable_date() -> str:
    return datetime.now().strftime("%a %b %d, %Y")


async def async_noop() -> None:
    pass


async def run_safe[T](
    cb: Callable[..., typing.Awaitable[T]], msg: str, **kwargs
) -> T | str:
    try:
        return await cb(**kwargs)
    except Exception as e:
        return f"{msg} {str(e)}"
