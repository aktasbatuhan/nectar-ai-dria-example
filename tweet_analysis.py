# tweet_analysis.py
from typing import List
from dria import SingletonTemplate
from dria.models import TaskResult
from pydantic import BaseModel, Field
from dria_workflows import *
from dria.factory.utilities import get_tags, parse_json, get_abs_path

class TweetAnalysisOutput(BaseModel):
    tweet: str = Field(..., description="Original Tweet")
    topic: str = Field(..., description="Topic Analysis")
    sentiment: str = Field(..., description="Tweet Sentiment")
    key_concepts: List[str] = Field(..., description="Key Concepts")

class TweetAnalysis(SingletonTemplate):
    tweet_text: str = Field(..., description="Tweet Content")
    
    OutputSchema = TweetAnalysisOutput

    def workflow(self):    
        builder = WorkflowBuilder(tweet_text=self.tweet_text)
        builder.set_max_tokens(800)
        builder.set_max_time(65)
        builder.set_max_steps(3)

        builder.generative_step(
            path=get_abs_path("analysis_prompt.md"),
            operator=Operator.GENERATION,
            outputs=[Write.new("output")],
        )

        flow = [Edge(source="0", target="_end")]
        builder.flow(flow)
        builder.set_return_value("output")
        return builder.build()
    
    def callback(self, result: List[TaskResult]) -> List[TweetAnalysisOutput]:
        results = []
        for r in result:
            # Parse the structured output from analysis
            analysis = parse_json(r.result.strip())
            results.append(
                TweetAnalysisOutput(
                    tweet=self.tweet_text,
                    topic=analysis['topic'],
                    sentiment=analysis['sentiment'],
                    key_concepts=analysis['key_concepts']
                )
            )
        return results

