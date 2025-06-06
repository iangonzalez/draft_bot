import time
import os
import asyncio
from dataclasses import dataclass


BATCH_SIZE = 32


@dataclass
class AutointerpResult:
    index: int
    prompt: str
    response: str


def create_unfilled_autointerp_results(prompts):
    """Create a list of AutointerpResult objects with the given prompts.

    The response field will be empty; the caller should fill these with fill_autointerp_results.
    """
    auto_interp_results = []
    for i, prompt in enumerate(prompts):
        if prompt is not None:
            auto_interp_results.append(AutointerpResult(index=i, prompt=prompt, response=None))
    return auto_interp_results


async def fill_autointerp_results(auto_interp_results, claude_client):
    batch_count = 0
    for i in range(0, len(auto_interp_results), BATCH_SIZE):
        print(f"[{time.strftime('%H:%M:%S')}] Processing batch {batch_count} of {BATCH_SIZE}")
        batch_count += 1
        batch = auto_interp_results[i:i + BATCH_SIZE]
        # Process the batch (send to Claude client)
        api_calls = []
        for result in batch:
            # Only create an api call if we havent processed this one yet.
            if result.response is None:
                api_calls.append(claude_client.get_response(result.prompt))
        async_results = await asyncio.gather(*api_calls)
        for result, async_result in zip(batch, async_results):
            if async_result is not None:
                result.response = async_result


def save_latent_autointerp_results(auto_interp_results, output_dir):
    """Save the autointerp results to a directory.

    The file naming convention assumes that the autointerp result index refers to a latent index.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write each result to a file
    for result in auto_interp_results:
        if result.response is not None:
            file_path = os.path.join(output_dir, f"latent_{result.index}.txt")
            with open(file_path, "w") as f:
                f.write(result.prompt)
                f.write("\n\n--------------------------- RESPONSE:\n\n")
                f.write(result.response)
        else:
            print(f"Skipping writing latent_{result.index}.txt as response is None.")

    print("Writing complete.")


def save_quiz_autointerp_results(auto_interp_results, answers, output_dir):
    """Save the autointerp results to a directory.

    The file naming convention assumes that the autointerp result index refers to the quiz question that the LLM was answering.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write each result to a file
    for result, answer in zip(auto_interp_results, answers):
        if result.response is not None:
            file_path = os.path.join(output_dir, f"quiz_result_sample_{result.index}.txt")
            with open(file_path, "w") as f:
                f.write(result.prompt)
                f.write("\n\n--------------------------- RESPONSE:\n\n")
                f.write(result.response)
                f.write(f"\n\n--------------------------- CORRECT ANSWER:\n\n<answer>{answer}</answer>")
        else:
            print(f"Skipping writing quiz_result_sample_{result.index}.txt as response is None.")

    print("Writing complete.")