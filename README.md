# 🧩 LLM NER(Named Entity Recognition) Benchmarks

<!--- BADGES: START --->

[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![GitHub - License](https://img.shields.io/github/license/Bae-ChangHyun/llm-structured-output-benchmarks?logo=github&style=flat&color=green)][#github-license]

![Github](https://img.shields.io/github/followers/Bae-ChangHyun?style=social)

[#github-license]: https://github.com/Bae-ChangHyun/llm-structured-output-benchmarks/blob/main/LICENSE

<!--- BADGES: END --->

Benchmark LLM on NER(Named Entity Recognition) tasks with various frameworks:
`Instructor`, `Mirascope`, `Langchain`, `LlamaIndex`, `Marvin`, `LMFormatEnforcer`, `etc`

## Attribution

This project is based on the excellent work of the original repository [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) created by `stephenleo`.
I would like to express gratitude to the original author for their contribution to the open source community.

## 🏆 NER Benchmark Results [2025-04-23]

| Framework                                                                                           |                Model                 | Reliability |  Latency p95 (s)   | Precision | Recall | F1 Score |
| --------------------------------------------------------------------------------------------------- | :----------------------------------: | :---------: | :----------------: | :-------: | :----: | :------: |
| [OpenAI Structured Output](https://github.com/openai/openai-python)                                 |        gpt-4o-mini-2024-07-18        |    1.000    |       3.459        |   0.834   | 0.748  |  0.789   |
| [LMFormatEnforcer](https://github.com/noamgat/lm-format-enforcer)                                   | unsloth/llama-3-8b-Instruct-bnb-4bit |    1.000    | 6.573<sup>\*</sup> |   0.701   | 0.262  |  0.382   |
| [Instructor](https://github.com/jxnl/instructor)                                                    |        gpt-4o-mini-2024-07-18        |    0.998    |       2.438        |   0.776   | 0.768  |  0.772   |
| [Mirascope](https://github.com/mirascope/mirascope)                                                 |        gpt-4o-mini-2024-07-18        |    0.989    |       3.879        |   0.768   | 0.738  |  0.752   |
| [Llamaindex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/openai_pydantic_program/) |        gpt-4o-mini-2024-07-18        |    0.979    |       5.771        |   0.792   | 0.310  |  0.446   |
| [Marvin](https://github.com/PrefectHQ/marvin)                                                       |        gpt-4o-mini-2024-07-18        |    0.979    |       3.270        |   0.822   | 0.776  |  0.798   |

<sup>\*</sup>GPU: `NVIDIA GeForce RTX 4080 Super`

## 🏃 Run the benchmark

1. Install the requirements using `pip install -r requirements.txt`
2. Set up your API keys:
   - Create a `.env` file in the root directory based on the provided `.env copy` template
   - Fill in your API keys:
     ```
     OLLAMA_HOST=your_ollama_host
     OPENAI_API_KEY=your_openai_key
     GOOGLE_API_KEY=your_google_key
     ```
   - Alternatively, you can export them directly: `export OPENAI_API_KEY=sk-...`
3. Run the benchmark using `python -m main run-benchmark`
   - You can specify a custom config file with `-c` or `--config`: `python -m main run-benchmark -c path/to/your/config.yaml`
4. Raw results are stored in the `results` directory.
5. Generate the results using `python -m main generate-results`
6. To get help on the command line arguments, add `--help` after the command. Eg., `python -m main run-benchmark --help`

## ⚙️ Configuring the benchmark

The benchmark is configured using a YAML file (default: `config.yaml`). Here's how to set up your configuration:

1. Each framework is defined as a top-level entry in the config file:

   ```yaml
   FrameworkName:
     - task: "ner" # Task type
       n_runs: 10 # Number of runs per sample
       init_kwargs: # Framework initialization parameters
         prompt: "Your prompt template with {text} placeholder"
         llm_model: "model-name"
         llm_model_family: "openai|google|ollama|transformers"
         source_data_pickle_path: "data/your_dataset.pkl"
         # Additional framework-specific parameters
   ```

2. Supported `llm_model_family` values:

   - `openai`: OpenAI models (requires OPENAI_API_KEY)
   - `google`: Google models like Gemini (requires GOOGLE_API_KEY)
   - `ollama`: Local models via Ollama (requires OLLAMA_HOST)
   - `transformers`: Hugging Face Transformers models

3. To add a new model configuration, simply create a new entry in the config file with appropriate parameters.

4. Example of adding a new model:
   ```yaml
   VanillaOpenAIFramework:
     - task: "ner"
       n_runs: 10
       init_kwargs:
         prompt: "Extract entities from: {text}"
         llm_model: "gpt-4o-mini"
         llm_model_family: "openai"
         source_data_pickle_path: "data/ner.pkl"
   ```

## 🧪 NER Benchmark methodology

- **Task**: Given a text, extract the entities present in it.
- **Data**:
  - Base data: [Synthetic PII Finance dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual)
  - Benchmarking test is run using a sampled data generated by running: `python -m data_sources.generate_dataset generate-ner-data`.
  - The data is sampled from the base data to achieve number of entities per row according to some distribution. See `python -m data_sources.generate_dataset generate-ner-data --help` for more details.
- **Prompt**: `Extract and resolve a list of entities from the following text: {text}`
- **Evaluation Metrics**:
  1. Reliability: The percentage of times the framework returns valid labels without errors. The average of all the rows `percent_successful` values.
  1. Latency: The 95th percentile of the time taken to run the framework on the data.
  1. Precision: The micro average of the precision of the framework on the data.
  1. Recall: The micro average of the recall of the framework on the data.
  1. F1 Score: The micro average of the F1 score of the framework on the data.
- **Experiment Details**: Run each row through the framework `n_runs` number of times and log the percent of successful runs for each row.

## 📊 Adding new data

1. Create a new pandas dataframe pickle file with the following columns:
   - `text`: The text to be sent to the framework
   - `labels`: List of labels associated with the text
   - See `data/multilabel_classification.pkl` for an example.
2. Add the path to the new pickle file in the `./config.yaml` file under the `source_data_pickle_path` key for all the frameworks you want to test.
3. Run the benchmark using `python -m main run-benchmark` to test the new data on all the frameworks!
4. Generate the results using `python -m main generate-results`

## 🏗️ Adding a new framework

The easiest way to create a new framework is to reference the `./frameworks/instructor_framework.py` file. Detailed steps are as follows:

1. Create a .py file in frameworks directory with the name of the framework. Eg., `instructor_framework.py` for the instructor framework.
2. In this .py file create a class that inherits `BaseFramework` from `frameworks.base`.
3. The class should define an `init` method that initializes the base class. Here are the arguments the base class expects:
   - `prompt` (str): Prompt template used. Obtained from the `init_kwargs` in the `./config.yaml` file.
   - `llm_model` (str): LLM model to be used. Obtained from the `init_kwargs` in the `./config.yaml` file.
   - `llm_model_family` (str): LLM model family to be used. Current supported values as `"openai"` and `"transformers"`. Obtained from the `init_kwargs` in the `./config.yaml` file.
   - `retries` (int): Number of retries for the framework. Default is $0$. Obtained from the `init_kwargs` in the `./config.yaml` file.
   - `source_data_picke_path` (str): Path to the source data pickle file. Obtained from the `init_kwargs` in the `./config.yaml` file.
   - `sample_rows` (int): Number of rows to sample from the source data. Useful for testing on a smaller subset of data. Default is $0$ which uses all rows in source_data_pickle_path for the benchmarking. Obtained from the `init_kwargs` in the `./config.yaml` file.
   - `response_model` (Any): The response model to be used. Internally passed by the benchmarking script.
4. The class should define a `run` method that takes three arguments:
   - `n_runs`: number of times to repeat each text
   - `expected_response`: Output expected from the framework. Use default value of `None`
   - `inputs`: a dictionary of `{"text": str}` where `str` is the text to be sent to the framework. Use default value of empty dictionary `{}`
5. This `run` method should create another `run_experiment` function that takes `inputs` as argument, runs that input through the framework and returns the output.
6. The `run_experiment` function should be annotated with the `@experiment` decorator from `frameworks.base` with `n_runs`, `expected_resposne` and `task` as arguments.
7. The `run` method should call the `run_experiment` function and return the four outputs `predictions`, `percent_successful`, `metrics` and `latencies`.
8. Import this new class in `frameworks/__init__.py`.
9. Add a new entry in the `./config.yaml` file with the name of the class as the key. The yaml entry can have the following fields
   - `n_runs`: number of times to repeat each text
   - `init_kwargs`: all the arguments that need to be passed to the `init` method of the class, including those mentioned in step 3 above.

## 🧭 Roadmap

1. Framework related tasks:
   | Framework | Named Entity Recognition |
   |-----------------------------------------------------------------------------------------------------|:--------------------------:|
   | [OpenAI Structured Output](https://github.com/openai/openai-python) | ✅ OpenAI |
   | [Instructor](https://github.com/jxnl/instructor) | ✅ OpenAI |
   | [Mirascope](https://github.com/mirascope/mirascope) | ✅ OpenAI |
   | [Marvin](https://github.com/PrefectHQ/marvin) | ✅ OpenAI |
   | [Llamaindex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/openai_pydantic_program/) | ✅ OpenAI |
   | [LM format enforcer](https://github.com/noamgat/lm-format-enforcer) | ✅ HF Transformers |
   | [Jsonformer](https://github.com/1rgs/jsonformer) | 💭 Planning |
   | [Guidance](https://github.com/guidance-ai/guidance) | 💭 Planning |
   | [DsPy](https://dspy-docs.vercel.app/docs/building-blocks/typed_predictors) | 💭 Planning |
   | [Langchain](https://python.langchain.com/v0.2/docs/tutorials/extraction/) | 💭 Planning |

## 💡 Contribution guidelines

Contributions are welcome! Here are the steps to contribute:

1. Please open an issue.

## 🙏 Feedback

If this work helped you in any way, please consider ⭐ this repository to give me feedback so I can spend more time on this project.
