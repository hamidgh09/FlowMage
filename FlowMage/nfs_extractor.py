import argparse
import os
import re
import shutil
from pathlib import Path

COMMENTS_REGEX: re.Pattern = re.compile(
    r'(?:\/\/(?:\\\n|[^\n])*\n)|(?:\/\*[\s\S]*?\*\/)|((?:R"([^(\\\s]{0,16})\([^)]*\)\2")|(?:@"[^"]*?")|'
    r'(?:"(?:\?\?\'|\\\\|\\"|\\\n|[^"])*?")|(?:\'(?:\\\\|\\\'|\\\n|[^\'])*?\'))'
)


def find_src_files(module_name: str, project_path: str, find_classes: bool = False) -> list:
    header_extensions = {'.h', '.hpp', '.hh'}
    source_extensions = {'.c', '.cpp', '.cxx', '.cc'}
    src_files = []

    class_def_regex = None
    member_func_regex = None
    if find_classes:
        class_def_regex = re.compile(r'\bclass\s+' + re.escape(module_name) + r'\b')
        member_func_regex = re.compile(re.escape(module_name) + r'::\w+')

    for file in (p.resolve() for p in Path(project_path).glob("**/*") \
                 if p.suffix in header_extensions | source_extensions):
        file_path = str(file)
        extension = file.suffix

        if extension in header_extensions | source_extensions:
            if find_classes:
                if is_searched_class(file_path, extension in header_extensions, class_def_regex, member_func_regex):
                    src_files.append(file_path)
            else:
                if is_searched_module(file_path, module_name, header_extensions | source_extensions):
                    src_files.append(file_path)

    return src_files


def is_searched_class(file_path: str, is_hdr: bool, class_def_regex: re.Pattern, member_func_regex: re.Pattern) -> bool:
    with open(file_path, 'r') as f:
        content = f.read()
        return (is_hdr and class_def_regex.search(content)) or (not is_hdr and member_func_regex.search(content))


def is_searched_module(file_path: str, module_name: str, extensions: set) -> bool:
    return any([os.path.basename(file_path) == module_name + ext for ext in extensions])


def find_fastclick_files(module_name: str, project_path: str, results_path: str) -> None:
    src_files = find_src_files(module_name, project_path, True)

    if src_files:
        os.makedirs(results_path, exist_ok=True)
        for file in src_files:
            copy_file_to_directory(file, results_path)

        # Fastclick Specific for flow elements
        if any([os.path.join("elements", "flow") in x for x in src_files]):
            copy_file_to_directory(
                os.path.join(project_path, "include", "click", "flow", "flowelement.hh"), results_path
            )

        if any([os.path.join("elements", "research") in x for x in src_files]):
            copy_file_to_directory(os.path.join(project_path, "elements", "research", "iflowmanager.hh"), results_path)


def find_vpp_files(module_name: str, project_path: str, results_path: str) -> None:
    src_files = find_src_files(module_name, project_path, False)

    if src_files:
        os.makedirs(results_path, exist_ok=True)
        for file in src_files:
            copy_file_to_directory(file, results_path)

        merge_files_in_directory(results_path, module_name)


def copy_file_to_directory(source_file: str, destination_directory: str) -> None:
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory, exist_ok=True)

    if source_file:
        shutil.copy(source_file, destination_directory)


def merge_files_in_directory(module_path: str, modules_name: str) -> None:
    merged_file_path = os.path.join(module_path, f"{modules_name}-merged.txt")

    if os.path.exists(merged_file_path):
        os.remove(merged_file_path)

    with open(merged_file_path, 'w') as merged_file:
        for root, _, files in os.walk(module_path):
            for file in files:
                if "merged" in file:
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    file_content = COMMENTS_REGEX.sub('', file_content)
                    merged_file.write(f"/*** {file} ***/\n\n")
                    merged_file.write(file_content + "\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--modules_names', type=str, required=True)
    parser.add_argument('--framework', choices=["fastclick", "vpp"], required=True, default="fastclick")
    parser.add_argument('--project_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, required=True)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    curr_path = os.path.dirname(__file__)

    project_path = os.path.abspath(os.path.join(curr_path, args.project_path))
    results_path = os.path.abspath(os.path.join(curr_path, args.results_path))

    modules_names = [x.strip() for x in args.modules_names.split(',')]
    for module_name in modules_names:
        if args.framework == "fastclick":
            find_fastclick_files(module_name, project_path, os.path.join(results_path, module_name))
        elif args.framework == "vpp":
            find_vpp_files(module_name, project_path, os.path.join(results_path, module_name))


if __name__ == "__main__":
    main(parse_args())
