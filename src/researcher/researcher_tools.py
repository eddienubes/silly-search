import asyncio
import logging
import typing
import json
from typing import Annotated, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from litellm import BaseModel
from config import cfg
from tavily_client import TavilyClient
import prompts
import utils


class SummaryOutputSchema(BaseModel):
    summary: str
    key_excerpts: str


@tool
async def search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """
    Fetch and summarize search results from Tavily search API.

    :param queries: List of search queries to execute
    :type queries: list[str]
    :return: Formatted string containing summarized search results
    :rtype: str
    """
    # TODO: Proper DI via Context
    # https://docs.langchain.com/oss/python/langchain/runtime
    tavily_client = TavilyClient(api_key=cfg.tavily_api_key)

    results = await tavily_client.search(
        queries=queries, max_results=max_results, topic=topic
    )

    summarization_tasks = [
        (
            utils.async_noop()
            if not result.get("raw_content")
            else summarize(content=result.get("raw_content"))
        )
        for result in results.values()
    ]

    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {
        url: {
            "title": result["title"],
            "content": result["content"] if not summary else summary,
        }
        for url, result, summary in zip(results.keys(), results.values(), summaries)
    }

    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    return json.dumps(summarized_results, separators=(",", ":"))


async def summarize(content: str) -> str:
    llm = (
        init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
        .with_structured_output(SummaryOutputSchema)
        .with_retry(stop_after_attempt=cfg.max_llm_retries)
    )

    content = content[: cfg.max_crawl_content_length]

    try:
        prompt = prompts.summarizer_prompt.format(
            webpage_content=content, date=utils.get_readable_date()
        )
        summary = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=cfg.summarization_timeout_sec,
        )
        summary = typing.cast(SummaryOutputSchema, summary)
        return f"""<summary>
        {summary.summary}
        </summary>

        <key_excerpts>
        {summary.key_excerpts}
        </key_excerpts>
        """
    except asyncio.TimeoutError:
        logging.warning(
            f"summarization failed with a timeout of {cfg.summarization_timeout_sec}, returning original content"
        )
        return content
    except Exception as e:
        logging.warning(
            f"Summarization failed due to error {str(e)}, returning original content"
        )
        return content
