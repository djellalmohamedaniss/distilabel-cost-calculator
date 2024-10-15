from pathlib import Path
from typing import Dict

import tiktoken
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from pydantic import BaseModel, Field, ValidationError, model_validator
from tiktoken import Encoding

from distilabel_cost_calculator.config.api_pricing_config import APIPricingConfig

# Load API pricing configuration
api_pricing_config_path = Path(__file__).parent / "config/api_pricing.yaml"
api_pricing_config = APIPricingConfig(api_pricing_config_path.resolve())


class GenerationCostCalculator(Step):
    api_model_name: str = Field(default_factory=str, exclude=True)
    api_pricing: dict | None = None

    @model_validator(mode="after")
    def check_pricing(self):
        if self.api_pricing is not None:
            required_keys = {"input_pricing", "output_pricing"}
            missing_keys = required_keys - self.api_pricing.keys()

            if missing_keys:
                raise ValueError(f"Missing required pricing keys: {missing_keys}")
        return self

    def _calculate_cost(self, distilabel_output: Dict, encoding: Encoding) -> Dict:
        """Calculate the total API cost for inputs and outputs based on pricing configuration."""
        api_pricing = self.api_pricing or api_pricing_config.get_model_by_name(
            self.api_model_name
        )

        input_cost = self._calculate_token_cost(
            distilabel_output, "raw_input", api_pricing["input_pricing"], encoding
        )
        output_cost = self._calculate_token_cost(
            distilabel_output, "raw_output", api_pricing["output_pricing"], encoding
        )

        total_cost = input_cost + output_cost
        return {
            "inputs": input_cost,
            "outputs": output_cost,
            "total_cost_str": f"{total_cost:.6f}$",
        }

    def _calculate_token_cost(
        self,
        distilabel_output: Dict,
        key_prefix: str,
        pricing: float,
        encoding: Encoding,
    ) -> float:
        """Generic method to calculate the token cost for input or output."""
        cost = 0
        for key, value in distilabel_output.items():
            if key_prefix in key:  # Look for either 'raw_input' or 'raw_output'
                if isinstance(
                    value, list
                ):  # Handles 'raw_input' where value is a list of messages
                    for message in value:
                        cost += (
                            len(encoding.encode(message["content"]))
                            * pricing
                            / 1_000_000
                        )
                else:  # Handles 'raw_output' where value is a single string
                    cost += len(encoding.encode(value)) * pricing / 1_000_000
                break
        return cost

    def process(self, inputs: StepInput) -> "StepOutput":
        """Process each input and calculate the associated cost."""
        encoding = tiktoken.encoding_for_model(model_name=self.api_model_name)
        for input_data in inputs:
            cost = self._calculate_cost(input_data["distilabel_metadata"], encoding)
            input_data["api_cost"] = cost
        yield inputs
