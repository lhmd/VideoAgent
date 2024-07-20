# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

PROMPT_NAME_TO_PATH = {
    "mmbench": """
You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}
    """,
    "mmbench-extra": """
You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer or any of the extra answers.
1 means that the response is completely different from the answer and all of the extra answers.

Example 1:
Question: Is it overcast?
Answer: no
Extra Answers: ['doesn't look like it', 'no',' it's sunny']
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Extra Answers: ['a woman', 'a lady', 'woman']
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Extra Answers: {extra_answers}
Response: {prediction}
    """,
}

from openai import AzureOpenAI

REGION = "eastus"
MODEL = "gpt-4o-2024-05-13"
API_KEY = "b47bf7405b388f3dcb00aa452a2aefdf"

API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
ENDPOINT = f"{API_BASE}/{REGION}"


def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())


def load_prompt(name: str):
    if name not in PROMPT_NAME_TO_PATH:
        raise ValueError("invalid prompt: {}".format(name))
    path = PROMPT_NAME_TO_PATH[name]
    return path
    # with path.open("r") as f:
    #     return f.read().strip()


def set_openai_key(key: Optional[str] = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key


def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_api(
    messages: list,
    model: str = "gpt-4",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    # client = openai.OpenAI()
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-02-01",
        azure_endpoint=ENDPOINT,
    )
    print("Start calling openai api")
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        # seed=seed,
        # max_tokens=max_tokens,
        # temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    verbose: bool = False,
):
    if prediction is None:
        return 0

    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    try:
        set_openai_key(key=API_KEY)
        messages = prepare_openai_messages(
            prompt.format(
                question=question,
                answer=answer,
                prediction=prediction,
                extra_answers=extra_answers,
            ),
        )
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            verbose=verbose,
        )
        print("output: ", output)
        return parse_score(output)
    except Exception as e:
        traceback.print_exc()
        raise e

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
        help="path to a results file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/metrics",
        help="path to an output directory (default: data/metrics)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="evaluate results even if responses are missing (default: false)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print verbose outputs (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only evaluate the first 5 questions",
    )
    args = parser.parse_args()
    assert args.results.exists()
    assert args.dataset.exists()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.results.stem + "-metrics.json")
    if args.verbose:
        print("output path: {}".format(args.output_path))
    return args


def main(args: argparse.Namespace):
    # load dataset
    dataset = json.load(args.dataset.open("r"))
    dataset = dataset[120:]
    dataset_question_ids = [item["question_id"] for item in dataset]
    question_id_to_item = {item["question_id"]: item for item in dataset}
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = json.load(args.results.open("r"))
    results_question_ids = [item["question_id"] for item in results]
    question_id_to_result = {result["question_id"]: result for result in results}
    print("found {:,} results".format(len(results)))

    # __import__("ipdb").set_trace()
    print("Len of dataset_question_ids: ", set(dataset_question_ids))
    print("Len of results_question_ids: ", set(results_question_ids))
    # check that results and dataset match
    if not args.force:
        assert len(dataset_question_ids) == len(results_question_ids)
        assert set(dataset_question_ids) == set(results_question_ids)

    # load scores
    all_scores = {}
    if args.output_path.exists():
        all_scores = json.load(args.output_path.open("r"))
        print("found {:,} existing scores".format(len(all_scores)))

    all_scannet_scores = {}
    all_hm3d_scores = {}
    # evaluate predictions
    for idx, question_id in enumerate(tqdm(results_question_ids)):
        if args.dry_run and idx >= 5:
            break

        item = question_id_to_item[question_id]
        if question_id not in all_scores:
            result = question_id_to_result[question_id]
            extra_answers = item["extra_answers"] if "extra_answers" in item else None

            # pre-process answers
            if result["answer"]:
                # remove anything after the last period
                end_idx = result["answer"].rfind(".")
                if end_idx >= 0 and end_idx + 1 < len(result["answer"]):
                    result["answer"] = result["answer"][: end_idx + 1]

            score = get_llm_match_score(
                question=item["question"],
                answer=item["answer"],
                prediction=result["answer"],
                extra_answers=extra_answers,
            )

            all_scores[question_id] = score
            json.dump(all_scores, args.output_path.open("w"), indent=2)

        else:
            score = all_scores[question_id]

        if 'scannet' in item['episode_history']:
            all_scannet_scores[question_id] = score
        elif 'hm3d' in item['episode_history']:
            all_hm3d_scores[question_id] = score
        else:
            raise NotImplementedError

    # calculate final score
    scores = np.array(list(all_scores.values()))
    scores = 100.0 * (np.clip(scores, 1, 5) - 1) / 4

    scannet_scores = np.array(list(all_scannet_scores.values()))
    scannet_scores = 100.0 * (np.clip(scannet_scores, 1, 5) - 1) / 4

    hm3d_scores = np.array(list(all_hm3d_scores.values()))
    hm3d_scores = 100.0 * (np.clip(hm3d_scores, 1, 5) - 1) / 4

    print("final score: {:.1f}".format(np.mean(scores)))
    print("final scannet score: {:.1f}".format(np.mean(scannet_scores)))
    print("final hm3d score: {:.1f}".format(np.mean(hm3d_scores)))
    
    #将score写到result.txt中
    # with open('result.txt', 'w') as f:
    #     f.write("final score: {:.1f}\n".format(np.mean(scores)))
    #     f.write("final scannet score: {:.1f}\n".format(np.mean(scannet_scores)))
    #     f.write("final hm3d score: {:.1f}\n".format(np.mean(hm3d_scores)))


if __name__ == "__main__":
    main(parse_args())