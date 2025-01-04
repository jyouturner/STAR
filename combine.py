import os
import json
from pathlib import Path
from typing import List, Dict

EXCLUDED_DIRS = {
    '.venv', 'venv', 'env',
    'node_modules',
    '__pycache__',
    '.git',
    'build', 'dist',
    'data',
    '.pytest_cache',
    '.mypy_cache',
    '.coverage',
    'htmlcov'
}

EXCLUDED_FILES = {
    '.pyc', '.pyo', '.pyd',
    '.so', '.dll', '.dylib',
    '.coverage', '.pytest_cache',
    '.DS_Store', '.env',
    '.gitignore',
    '.md'
    'consolidated_project.md',
    os.path.basename(__file__)
}

def should_process_path(path: str) -> bool:
    parts = Path(path).parts
    return not any(part.startswith('.') or part in EXCLUDED_DIRS for part in parts)

def collect_project_files(root_dir: str) -> Dict[str, str]:
    project_files = {}
    important_files = ['pyproject.toml', 'requirements.txt']
    
    for root, dirs, files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('.')]
        
        if not should_process_path(root):
            continue
            
        for file in files:
            if any(file.endswith(ext) for ext in EXCLUDED_FILES) or file in EXCLUDED_FILES:
                continue
                
            if file.endswith(('.py', '.toml', '.yaml', '.yml')) or file in important_files:
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        relative_path = str(file_path.relative_to(root_dir))
                        project_files[relative_path] = content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return project_files

def create_consolidated_output(project_files: Dict[str, str]) -> str:
    output = "# Consolidated Poetry Project Files\n\n"
    
    output += "## Project Structure\n\n"
    output += "```\n"
    for filepath in sorted(project_files.keys()):
        output += f"{filepath}\n"
    output += "```\n\n"
    
    output += "## File Contents\n\n"
    for filepath, content in sorted(project_files.items()):
        output += f"### {filepath}\n\n"
        output += "```"
        if filepath.endswith('.py'):
            output += "python"
        elif filepath.endswith('.toml'):
            output += "toml"
        output += "\n"
        output += content
        output += "\n```\n\n"
    
    return output

def main():
    root_dir = "."
    project_files = collect_project_files(root_dir)
    
    output = create_consolidated_output(project_files)
    
    output_file = "consolidated_project.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"Project files consolidated into {output_file}")
    print(f"Total files processed: {len(project_files)}")

if __name__ == "__main__":
    main()