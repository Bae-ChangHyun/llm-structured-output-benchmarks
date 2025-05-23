from typing import List, Any

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from frameworks.base import BaseFramework, experiment

def clean_response(response):
    """
    If the answer contains <think> tags, removes the content wrapped in <think></think> tags
    and returns only the content after the tags.
    
    Args:
        answer: String that might contain <think> tags
    
    Returns:
        Cleaned answer with thinking part removed
    """
    if "<think>" in response and "</think>" in response:
        # Find the closing tag position
        end_tag_pos = response.find("</think>")
        if end_tag_pos != -1:
            # Return everything after the </think> tag
            return response[end_tag_pos + len("</think>"):].strip()
    
    # If no tags or improper tag format, return original answer
    return response

class ThinkTagRemover(BaseOutputParser):
    def parse(self, text):
        return clean_response(text)


class LangchainParserFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.llm_provider == "openai":
            self.llm = ChatOpenAI(model=self.llm_model)
        elif self.llm_provider == "ollama":
            self.llm = ChatOllama(model=self.llm_model,
                                  base_url=self.base_url,
                                 api_key="ollama")
        elif self.llm_provider == "vllm":
            self.llm = ChatOpenAI(model=self.llm_model,
                                  base_url=self.base_url,
                                  api_key="vllm")
        elif self.llm_provider == "google":
            self.llm = ChatGoogleGenerativeAI(model=self.llm_model)

        self.parser = ThinkTagRemover()
        self.parser2 = PydanticOutputParser(pydantic_object=self.response_model)
        

    def run(
        self, max_tries: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(max_tries=max_tries, expected_response=expected_response)
        def run_experiment(inputs):
            
            prompt = ChatPromptTemplate.from_messages([
                ("user", self.prompt + "\n{format_instructions}")
            ])
            
            prompt = prompt.partial(format_instructions=self.parser2.get_format_instructions())
        
            chain = prompt | self.llm | self.parser | self.parser2
            
            response = chain.invoke(inputs)
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies