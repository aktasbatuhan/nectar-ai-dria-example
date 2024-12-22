# run_pipeline.py
from dria import DriaDataset, DatasetGenerator, Model
from tweet_analysis import TweetAnalysis
from prompt_generation import PromptGeneration
import asyncio
import pandas as pd
import json
import logging

async def run_pipeline(tweets_file_path: str):
    # Load tweets from CSV
    df = pd.read_csv(tweets_file_path)
    
    # Prepare instructions for first stage (tweet analysis)
    analysis_instructions = [
        {"tweet_text": tweet} for tweet in df['text'].tolist()
    ]

    # Initialize dataset for final output
    final_dataset = DriaDataset(
        name="Nectar_AI_Dria_Demo",
        description="Generated prompts based on tweet analysis",
        schema=PromptGeneration.OutputSchema
    )

    # Initialize generator
    generator = DatasetGenerator(dataset=final_dataset,log_level=logging.DEBUG)

    # Run the pipeline
    await generator.generate(
        instructions=analysis_instructions,
        singletons=[TweetAnalysis, PromptGeneration],
        models=[Model.GPT4O, Model.GPT4O_MINI, Model.GEMINI_15_FLASH]
    )

    # Export results
    final_dataset.to_json("generated_prompts.json")
    return final_dataset.to_pandas()

if __name__ == "__main__":
    tweets_file = ""###Your CSV path  
    asyncio.run(run_pipeline(tweets_file))