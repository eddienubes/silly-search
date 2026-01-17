import asyncio
from typing import Literal
import tavily


class TavilyClient:
    def __init__(self, api_key: str) -> None:
        self.__client = tavily.AsyncTavilyClient(api_key=api_key)

    async def search(
        self,
        queries: list[str],
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> dict:
        """
        https://docs.tavily.com/documentation/api-reference/endpoint/search

        :return: dict[url] = {*tavily_result}
        :rtype: dict
        """

        search_tasks = [
            self.__client.search(
                query=query,
                max_results=max_results,
                topic=topic,
                include_raw_content=True,
            )
            for query in queries
        ]

        responses = await asyncio.gather(*search_tasks)

        unique_results = {}
        for response in responses:
            for result in response["result"]:
                if result["url"] not in unique_results:
                    unique_results["url"] = {**result, "query": response["query"]}

        return unique_results
