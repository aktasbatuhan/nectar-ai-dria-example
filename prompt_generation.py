from typing import List
from dria import SingletonTemplate
from dria.models import TaskResult
from pydantic import BaseModel, Field
from dria_workflows import *
from dria.factory.utilities import get_tags, parse_json, get_abs_path

# prompt_generation.py
class PromptOutput(BaseModel):
    original_tweet: str = Field(..., description="Original Tweet")
    topic: str = Field(..., description="Topic")
    generated_prompt: str = Field(..., description="Generated Prompt")

class PromptGeneration(SingletonTemplate):
    tweet: str = Field(..., description="Original Tweet")
    topic: str = Field(..., description="Topic")
    key_concepts: List[str] = Field(..., description="Key Concepts")
    
    OutputSchema = PromptOutput

    def workflow(self):    
        builder = WorkflowBuilder(
            tweet=self.tweet,
            topic=self.topic,
            key_concepts=self.key_concepts
        )
        builder.set_max_tokens(800)
        builder.set_max_time(65)
        builder.set_max_steps(3)

        builder.generative_step(
            path=get_abs_path("prompt_generation.md"),
            operator=Operator.GENERATION,
            outputs=[Write.new("output")],
        )

        flow = [Edge(source="0", target="_end")]
        builder.flow(flow)
        builder.set_return_value("output")
        return builder.build()
    
    def callback(self, result: List[TaskResult]) -> List[PromptOutput]:
        results = []
        for r in result:
            results.append(
                PromptOutput(
                    original_tweet=self.tweet,
                    topic=self.topic,
                    generated_prompt=r.result.strip()
                )
            )
        return results