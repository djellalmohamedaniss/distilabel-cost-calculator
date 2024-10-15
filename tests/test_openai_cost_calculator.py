import pytest
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextClassification, TextGeneration

from distilabel_cost_calculator import GenerationCostCalculator

OPENAI_API_KEY = "XXX"


def test_cost_calculator_with_generation_pipeline():

    with Pipeline("test_openai_cost_calculator") as pipeline:
        model = OpenAILLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        data = LoadDataFromDicts(data=[{"instruction": "Generate me a quote."}])
        task = TextGeneration(name="text-generation", llm=model, num_generations=2)
        cost_step = GenerationCostCalculator(api_model_name="gpt-4o-mini")
        data >> task >> cost_step
    pipeline.run(use_cache=False)


def test_cost_calculator_with_classification_pipeline():

    with Pipeline("test_openai_cost_calculator_classification") as pipeline:
        model = OpenAILLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        data = LoadDataFromDicts(data=[{"text": "I really liked the food!"}] * 5)
        task = TextClassification(
            context="You are an AI assisant that classifies text.",
            available_labels=["positive", "negative"],
            n=1,
            llm=model,
        )
        cost_step = GenerationCostCalculator(api_model_name="gpt-4o-mini")
        data >> task >> cost_step
    pipeline.run(use_cache=False)


def test_cost_calculator_with_generation_pipeline_and_dynamic_pricing():

    with Pipeline("test_openai_cost_calculator") as pipeline:
        model = OpenAILLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        data = LoadDataFromDicts(data=[{"instruction": "Generate me a quote."}])
        task = TextGeneration(name="text-generation", llm=model, num_generations=2)
        cost_step = GenerationCostCalculator(
            api_model_name="gpt-4o-mini",
            api_pricing={"input_pricing": 0.15, "output_pricing": 0.6},
        )
        data >> task >> cost_step
    pipeline.run(use_cache=False)
