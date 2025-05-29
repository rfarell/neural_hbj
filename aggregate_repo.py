#!/usr/bin/env python3
"""
Aggregates repository contents into a single text file for LLM context.
Respects .gitignore and provides file size/count summary.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Set, Tuple
import fnmatch

DEFAULT_OUTPUT = "tmp/aggregate_repo.txt"
MAX_FILE_SIZE = 100 * 1024  # 100KB default

# File extensions to include
INCLUDE_PATTERNS = {
    "*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.java", "*.cpp", "*.c", "*.h",
    "*.go", "*.rs", "*.sql", "*.graphql", "*.proto", "*.yaml", "*.yml",
    "*.json", "*.toml", "*.ini", "*.md", "*.rst", "*.txt", "*.sh", "*.bash",
    "Dockerfile*", "Makefile", "*requirements*.txt", "pyproject.toml",
    "package*.json", ".env.example", "setup.py", "setup.cfg"
}

# Patterns to exclude
EXCLUDE_PATTERNS = {
    "__pycache__", ".pytest_cache", "node_modules", ".next", "dist", "build",
    "target", ".git", ".svn", ".vscode", ".idea", "*.swp", "*.log", "logs",
    "*.tmp", ".DS_Store", "venv", "env", ".venv", ".conda", "*.pyc", "*.pyo",
    "*.egg-info", "*.egg", ".coverage", "htmlcov", ".tox", "*.so", "*.dylib",
    "*.dll", "*.exe", "*.bin", "tmp", "temp", ".mypy_cache", ".ruff_cache"
}


def read_gitignore(repo_root: Path) -> Set[str]:
    """Read .gitignore patterns."""
    patterns = set()
    gitignore_path = repo_root / ".gitignore"
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.add(line)
    
    # Also check for .aggregate_ignore
    aggregate_ignore = repo_root / ".aggregate_ignore"
    if aggregate_ignore.exists():
        with open(aggregate_ignore, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.add(line)
    
    return patterns | EXCLUDE_PATTERNS


def should_include_file(file_path: Path, include_patterns: Set[str], 
                       exclude_patterns: Set[str]) -> bool:
    """Check if file should be included based on patterns."""
    file_str = str(file_path)
    
    # Check exclude patterns first
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(file_str, pattern) or pattern in file_str:
            return False
    
    # Check include patterns
    for pattern in include_patterns:
        if fnmatch.fnmatch(file_path.name, pattern):
            return True
    
    return False


def is_binary(file_path: Path) -> bool:
    """Check if file is binary."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
            # Check for high ratio of non-text bytes
            text_chars = set(range(32, 127)) | {9, 10, 13}
            non_text = sum(1 for b in chunk if b not in text_chars)
            return non_text / len(chunk) > 0.3
    except:
        return True


def generate_tree(repo_root: Path, exclude_patterns: Set[str]) -> str:
    """Generate directory tree structure."""
    tree_lines = []
    
    def add_tree(path: Path, prefix: str = "", is_last: bool = True):
        if path.name in exclude_patterns:
            return
            
        relative = path.relative_to(repo_root) if path != repo_root else Path(".")
        
        if path == repo_root:
            tree_lines.append(str(relative))
        else:
            connector = "└── " if is_last else "├── "
            tree_lines.append(prefix + connector + path.name)
        
        if path.is_dir():
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            items = [item for item in items if item.name not in exclude_patterns]
            
            for i, item in enumerate(items):
                extension = "    " if is_last else "│   "
                add_tree(item, prefix + extension, i == len(items) - 1)
    
    add_tree(repo_root)
    return "\n".join(tree_lines)


def aggregate_repository(repo_root: Path, output_file: Path, 
                        max_size: int) -> Tuple[int, int]:
    """Aggregate repository contents into single file."""
    exclude_patterns = read_gitignore(repo_root)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    total_size = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # Write directory tree
        out.write("DIRECTORY STRUCTURE:\n")
        out.write("=" * 50 + "\n")
        out.write(generate_tree(repo_root, exclude_patterns))
        out.write("\n\n")
        
        # Process files
        for root, dirs, files in os.walk(repo_root):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns]
            
            root_path = Path(root)
            
            for file in sorted(files):
                file_path = root_path / file
                relative_path = file_path.relative_to(repo_root)
                
                if not should_include_file(file_path, INCLUDE_PATTERNS, exclude_patterns):
                    continue
                
                if file_path.stat().st_size > max_size:
                    continue
                
                if is_binary(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_size = len(content.encode('utf-8'))
                    out.write(f"\n----- {relative_path} ({file_size} bytes) -----\n")
                    out.write(content)
                    if not content.endswith('\n'):
                        out.write('\n')
                    out.write("----- END -----\n")
                    
                    file_count += 1
                    total_size += file_size
                    
                except Exception as e:
                    print(f"Warning: Could not read {relative_path}: {e}", file=sys.stderr)
    
    return file_count, total_size


def main():
    parser = argparse.ArgumentParser(description="Aggregate repository for LLM context")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                       help=f"Output file path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--max-size", type=int, default=MAX_FILE_SIZE,
                       help=f"Maximum file size in bytes (default: {MAX_FILE_SIZE})")
    
    args = parser.parse_args()
    
    repo_root = Path.cwd()
    output_file = Path(args.output)
    
    print(f"Aggregating repository: {repo_root}")
    print(f"Output file: {output_file}")
    
    file_count, total_size = aggregate_repository(repo_root, output_file, args.max_size)
    
    print(f"\n✅ Aggregation complete!")
    print(f"Files included: {file_count}")
    print(f"Total size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()