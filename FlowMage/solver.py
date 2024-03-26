import argparse
import os
import json
import sys
import re

from FlowMage.optimizer import Optimizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=["fastclick", "vpp"], required=False, default="fastclick")
    parser.add_argument('--llm_analysis_path', type=str, required=False, default="./llm-results/fc-vpp-merged-gpt4.json")
    parser.add_argument('--input', type=str, required=False, default="./sample.click")
    parser.add_argument('--output', type=str, required=False, default="./output.click")

    return parser.parse_args()


def parse_llm_analysis(llm_analysis_path: str) -> dict:
    if not os.path.exists(llm_analysis_path):
        print(f"File {llm_analysis_path} does not exist")
        return {}

    with open(llm_analysis_path, "r") as f:
        return json.load(f)


def parse_input_file(input_path: str) -> dict:
    """
    Receives a click configuration file and return a dictionary with the existing elements and arguments
    """
    if not os.path.exists(input_path):
        print(f"File {input_path} does not exist")
        return {}

    pattern = r'\b\w*\s*::\s*(\w+)\s*(\(.*?\))?'
    with open(input_path, "r") as f:
        content = f.read()
    f.close()

    matches = re.findall(pattern, content)
    elements = {element: args.strip('()') if args else None for element, args in matches}

    return elements

def isStateful(llm_analysis: dict, element: str) -> bool:
    if (llm_analysis.get(element) is None):
        return False
    results = llm_analysis[element]
    stateful_count = sum(1 for item in results.values() if item['merged']['result']['statefulness'] == 'stateful')
    if (stateful_count > len(results)/2):
        return True
    return False

def run_optimizer(all_elements: dict, stateful_elements: list) -> dict:
    """
    Receives a dictionary with all elements and their arguments and a list with the stateful elements
    and returns a dictionary with the optimized elements and their arguments
    """
    optimizer = Optimizer(all_elements, stateful_elements)
    return optimizer.optimize()

def optimize_elements(input_path: str, llm_analysis: dict) -> dict:
    all_elements = parse_input_file(input_path)
    stateful_elements = dict(filter(lambda item: isStateful(llm_analysis, str(item[0])), all_elements.items()))
    optimized_elements = run_optimizer(all_elements, list(stateful_elements.keys()))
    return stateful_elements


def main(args: argparse.Namespace):
    
    if not args.framework == "fastclick":
        print("Framework not supported")
        return 1
    
    curr_path = os.path.dirname(__file__)
    input_file = os.path.abspath(os.path.join(curr_path, args.input))
    llm_analysis_file = os.path.abspath(os.path.join(curr_path, args.llm_analysis_path))

    optimize_elements(input_file, parse_llm_analysis(llm_analysis_file))

    return 0

if __name__ == "__main__":
    main(parse_args())
