# Description

A custom step for LLM API cost calculation for the **distilabel** library. This library allows you to calculate API usage costs based on token consumption and supports customizable pricing models for different LLMs.

## Requirements

- Python: `^3.10.15`
- Distilabel: `>=1.3.2`

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

    # You can provide custom pricing for the API model
    custom_pricing = {
        "input_pricing": 0.15,  # Custom price per token for inputs
        "output_pricing": 0.6  # Custom price per token for outputs
    }

    cost_step = GenerationCostCalculator(
        api_model_name="gpt-4o-mini",
        api_pricing=custom_pricing  # Optional: Override default pricing
    )

    data >> task >> cost_step
pipeline.run(use_cache=False)

# API cost details will be found in the 'api_cost' column in the Distiset.
# 'api_cost': {'inputs': 7.5e-07, 'outputs': 1.019e-05, 'total_cost_str': '0.000011$'}
```
