## Description

A custom Step for LLM API cost calculation for the distilabel library.

## Installation

```shell
pip install distilabel-cost-calculator
```

## Example

```python
from distilabel_cost_calculator import GenerationCostCalculator

with Pipeline("data_generation_with_cost_calculator") as pipeline:
        model = OpenAILLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        data = LoadDataFromDicts(data=[{"instruction": "Generate me a quote."}])
        task = TextGeneration(name="text-generation", llm=model, num_generations=2)
        cost_step = GenerationCostCalculator(api_model_name="gpt-4o-mini")
        data >> task >> cost_step
    pipeline.run(use_cache=False)

    """
    API cost details will be found in the 'api_cost' column in the Distiset.
    'api_cost': {'inputs': 7.5e-07, 'outputs': 1.0199999999999999e-05, 'total_cost_str': '0.000011$'}
    """
```
