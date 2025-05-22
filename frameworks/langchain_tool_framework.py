import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 


from frameworks.base import BaseFramework, experiment
from typing import Any

class LangchainToolFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if self.llm_provider == "openai":
            self.llm = ChatOpenAI(model=self.llm_model)
        elif self.llm_provider == "ollama" or self.llm_provider == "vllm":
            self.llm = ChatOpenAI(model=self.llm_model,
                                  base_url=self.base_url,
                                  api_key="empty")
        elif self.llm_provider == "google":
            self.llm = ChatOpenAI(model=self.llm_model,
                                  base_url=self.base_url,
                                  api_key=os.environ.get("GOOGLE_API_KEY"))

        self.structured_llm = self.llm.with_structured_output(self.response_model)

    def run(
        self, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):

            prompt = ChatPromptTemplate.from_messages([
                ("user", self.prompt)
            ])

            chain = prompt | self.structured_llm

            response = chain.invoke(inputs) 
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies