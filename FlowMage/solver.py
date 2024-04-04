import argparse
import os
import json
import sys
import re

from collections import Counter
from optimizer import Optimizer

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
        result = json.load(f)
        result = {key: aggregate_all_results(value) for key, value in result.items()}
    f.close()
    return result


def aggregate_all_results(nf_result: dict) -> dict:
    statefulness = aggregate_statefulness(nf_result)
    if statefulness == 'stateless':
        return {
            "stateful": 'stateless',
        }

    intensity = aggregate_intensity(nf_result)
    flow_def = aggregate_flow_def(nf_result)
    pointer_chasing = aggregate_pointer_chasing(nf_result)
    return {
        "stateful": 'stateful',
        "intensity": intensity,
        "flow_def": flow_def,
        "pointer_chasing": pointer_chasing,
        }

def aggregate_pointer_chasing(nf_result: dict) -> bool:
    total = 0
    pointer_chasing = 0
    for result in nf_result.values():
        total += 1
        try:
            if result.get('merged',{}).get('result',{}).get('pointer', False) == True:
                pointer_chasing += 1
            else:
                size_list = result.get('merged',{}).get('result',{}).get('size', [])
                if size_list == None:
                    continue
                has_pointer = size_list[1] if len(size_list) > 1 else False
                if has_pointer:
                    pointer_chasing += 1
        except KeyError:
            continue
    return pointer_chasing >= total/2

def aggregate_flow_def(nf_result: dict) -> list:
    total = 0
    definitions_list = []
    for result in nf_result.values():
        total += 1
        try:
            definitions_list.append(result['merged']['result']['key'])
        except KeyError:
            continue
    
    # Flatten the list to count each item separately
    flattened_list = [item for sublist in definitions_list for item in sublist]
    # Count occurrences of each item
    count = Counter(flattened_list)
    most_common_items = [item for item, occurrence in count.items() if occurrence >= total/2]
    return most_common_items

def aggregate_intensity(nf_result: dict) -> str:
    total = 0
    per_packet = 0
    for result in nf_result.values():
        total += 1
        try:
            if result['merged']['result']['intensity'] == 'per-packet':
                per_packet += 1
        except KeyError:
            continue
    return 'per-packet' if per_packet > total/2 else 'per-flow'

def aggregate_statefulness(nf_result: dict) -> str:
    total = 0
    stateful = 0
    for result in nf_result.values():
        total += 1
        try:
            if result['merged']['result']['statefulness'] == 'stateful':
                stateful += 1
        except KeyError:
            continue
    return 'stateful' if stateful > total/2 else 'stateless'

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
    return True if llm_analysis[element].get('stateful', 'stateless') == 'stateful' else False

def run_optimizer(llm_analysis: dict, stateful_elements: dict) -> dict:
    """
    Receives a dictionary with all elements and their arguments and a list with the stateful elements
    and returns a dictionary with the optimized elements and their arguments
    """
    optimizer = Optimizer(llm_analysis, stateful_elements)
    return optimizer.optimize()

def optimize_elements(all_elements: dict, llm_analysis: dict) -> dict:
    stateful_elements = dict(filter(lambda item: isStateful(llm_analysis, str(item[0])), all_elements.items()))
    optimized_elements = run_optimizer(llm_analysis, stateful_elements)
    all_elements.update(optimized_elements)
    return all_elements

def write_output(optimized_elements: dict, input_path: str, output_path: str) -> None:
    # Open and read the input file
    with open(input_path, 'r') as file:
        content = file.readlines()
    
    # Iterate through each line of the file
    for i, line in enumerate(content):
        # Check if any key in the dictionary is in the current line
        for key, value in optimized_elements.items():
            # If the key is found in the line, replace the values
            if key in line:
                # Extract the substring before the first parenthesis and add the new values
                updated_line = line.split('(', 1)[0] + '(' + value + ')\n'
                # Update the line in the content
                content[i] = updated_line
                break
    
    # Write the updated content to the output file
    with open(output_path, 'w') as file:
        file.writelines(content)

def main(args: argparse.Namespace):
    
    if not args.framework == "fastclick":
        print("Framework not supported")
        return 1
    
    curr_path = os.path.dirname(__file__)
    input_file = os.path.abspath(os.path.join(curr_path, args.input))
    llm_analysis_file = os.path.abspath(os.path.join(curr_path, args.llm_analysis_path))
    
    all_elements = parse_input_file(input_file)
    llm_analysis = parse_llm_analysis(llm_analysis_file)
    optimized_elements = optimize_elements(all_elements, llm_analysis)

    output_path = os.path.abspath(os.path.join(curr_path, args.output))
    write_output(optimized_elements, input_file, output_path)
    print(f"Optimized configuration written to {output_path}")
    return 0

if __name__ == "__main__":
    main(parse_args())
