import logging
import time
from tavily import TavilyClient
from groq import Groq
from app.prompt_template import web_prompt_template
from app.config import CONFIG
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

load_dotenv()

class WebSearchAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search_web(self, query: str) -> dict:
        """
        Performs a web search for the given query using the Tavily API and generates an answer using a language model.

        Args:
            query (str): The search query.

        Returns:
            dict: A dictionary containing the generated answer, extracted content as ground truth, and source URLs.
        """

        start_time = time.time()
        logger.info(f"Web search for query: {query}")

        try:
            response = self.tavily.search(query=query, max_results=CONFIG["web_search"]["max_results"])
            results = response.get("results", [])
            logger.info(f"Retrieved {len(results)} web results")

            if not results:
                logger.warning("No web results found for query.")
                return {
                    "answer": "I couldn’t find a definitive answer based on available web information.",
                    "ground_truth": [],
                    "sources": []
                }

            web_content = []
            sources = []
            for result in results:
                content = result.get("content", "")
                url = result.get("url", "unknown")
                if content:
                    web_content.append(content[:2000])
                    sources.append(url)
                    logger.info(f"Extracted {len(content)} chars from {url}")
                else:
                    logger.warning(f"No content in result: {url}")

            if not web_content:
                logger.warning("No valid content extracted from web.")
                return {
                    "answer": "I couldn’t find a definitive answer based on available web information.",
                    "ground_truth": [],
                    "sources": sources
                }

            combined_content = "\n".join(web_content)

            formatted_prompt = web_prompt_template.format(
                web_content=combined_content, query=query
            )

            logger.info(f"Sending web prompt to Groq API, length: {len(formatted_prompt)}")
            response = self.client.chat.completions.create(
                model=CONFIG["groq"]["model"],
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=CONFIG["groq"]["temperature"],
                max_tokens=CONFIG["groq"]["max_tokens"]
            )

            answer = response.choices[0].message.content
            end_time = time.time()
            logger.info(f"Web Search Time: {end_time - start_time:.2f} sec")

            return {
                "answer": answer,
                "ground_truth": web_content,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Web search failed: {str(e)}", exc_info=True)
            return {
                "answer": "I couldn’t find a definitive answer based on available web information.",
                "ground_truth": [],
                "sources": []
            }