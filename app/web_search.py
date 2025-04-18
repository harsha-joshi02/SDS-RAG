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

    def search_web(self, query: str) -> str:
        start_time = time.time()
        logger.info(f"Web search for query: {query}")

        try:
            response = self.tavily.search(query=query, max_results=CONFIG["web_search"]["max_results"])
            results = response.get("results", [])
            logger.info(f"Retrieved {len(results)} web results")
        
            if not results:
                logger.warning("No web results found for query.")
                return "I couldn’t find a definitive answer based on available web information."

            web_content = []
            for result in results:
                content = result.get("content", "")
                if content:
                    web_content.append(content[:2000]) 
                    logger.info(f"Extracted {len(content)} chars from {result.get('url', 'unknown')}")
                else:
                    logger.warning(f"No content in result: {result.get('url', 'unknown')}")

            if not web_content:
                logger.warning("No valid content extracted from web.")
                return "I couldn’t find a definitive answer based on available web information."

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

            return answer

        except Exception as e:
            logger.error(f"Web search failed: {str(e)}", exc_info=True)
            return "I couldn’t find a definitive answer based on available web information."