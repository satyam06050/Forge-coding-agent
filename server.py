from fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.tools.file_management.delete import DeleteFileTool
from langchain_community.tools.file_management.move import MoveFileTool
from langchain_community.tools.file_management.copy import CopyFileTool
from pathlib import Path
import subprocess
import re

load_dotenv()

mcp = FastMCP(name='server')


# ── File Read / Write ─────────────────────────────────────────────────────────

@mcp.tool
def read_file(path: str) -> str:
    """read the full content of a file at the given absolute path."""
    return ReadFileTool().run(path)


@mcp.tool
def write_file(path: str, content: str) -> str:
    """write content to a file (creates the file if it doesn't exist, overwrites if it does).
    always write the FULL file content — not just the changed portion."""
    WriteFileTool().run({"file_path": path, "text": content})
    return f"Done! Written to {path}"


@mcp.tool
def find_and_replace_in_file(path: str, old_string: str, new_string: str) -> str:
    """replace a specific string inside a file without rewriting the entire file.
    use this for targeted edits. use write_file only when you need to rewrite the whole file."""
    p = Path(path)
    if not p.exists():
        return f"Error: file '{path}' does not exist."
    content = p.read_text()
    if old_string not in content:
        return f"Error: string not found in '{path}'."
    p.write_text(content.replace(old_string, new_string))
    return f"Done! Replaced all occurrences in '{path}'."


@mcp.tool
def delete_file(path: str) -> str:
    """delete a file at the given path."""
    DeleteFileTool().run(path)
    return f"Done! '{path}' deleted."


@mcp.tool
def move_or_rename_file(source: str, destination: str) -> str:
    """move or rename a file from source path to destination path."""
    MoveFileTool().run({"source_path": source, "destination_path": destination})
    return f"Done! Moved '{source}' to '{destination}'"


@mcp.tool
def copy_file(source: str, destination: str) -> str:
    """copy a file from source path to destination path."""
    CopyFileTool().run({"source_path": source, "destination_path": destination})
    return f"Done! Copied '{source}' to '{destination}'"


@mcp.tool
def create_directory(path: str) -> str:
    """create a new directory including any missing parent directories."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return f"Done! Directory '{path}' created."


# ── Project Navigation ────────────────────────────────────────────────────────

@mcp.tool
def list_all_files_in_project(path: str) -> str:
    """returns a flat list of ALL file paths inside the given directory recursively.
    use this ONLY when you need to discover where a file is located or understand the full project layout.
    pass the PROJECT ROOT as path. do NOT use this just to find a known file — use search_in_files instead."""
    IGNORE = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    root = Path(path)
    result = []
    for item in sorted(root.rglob("*")):
        if any(part in IGNORE for part in item.parts):
            continue
        result.append(str(item.relative_to(root)))
    return "\n".join(result)


@mcp.tool
def search_in_files(directory: str, pattern: str, file_extension: str = "") -> str:
    """search for a filename, string, or regex pattern across all files in a directory.
    use this to locate a file by name or find where a function/variable is defined.
    optionally filter by file extension (e.g. '.py', '.js'). returns matching file paths and lines."""
    root = Path(directory)
    IGNORE = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    results = []
    regex = re.compile(pattern)
    for file in sorted(root.rglob("*")):
        if any(part in IGNORE for part in file.parts):
            continue
        if not file.is_file():
            continue
        if file_extension and file.suffix != file_extension:
            continue
        try:
            lines = file.read_text(errors="ignore").splitlines()
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append(f"{file}:{i}: {line.rstrip()}")
        except Exception:
            continue
    return "\n".join(results) if results else "No matches found."


@mcp.tool
def get_file_metadata(path: str) -> str:
    """get metadata of a file: size in bytes, line count, and last modified time."""
    p = Path(path)
    if not p.exists():
        return f"Error: file '{path}' does not exist."
    stat = p.stat()
    lines = len(p.read_text(errors="ignore").splitlines())
    return f"Path: {path}\nSize: {stat.st_size} bytes\nLines: {lines}\nLast modified: {stat.st_mtime}"


# ── Shell / Execution ─────────────────────────────────────────────────────────

@mcp.tool
def run_terminal_command(command: str, working_directory: str = "") -> str:
    """run any shell command and return stdout + stderr. use for running scripts, installing packages,
    running tests (pytest, npm test), building projects, or checking the environment.
    optionally pass a working_directory. commands timeout after 120 seconds."""
    cwd = working_directory if working_directory else None
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120, cwd=cwd)
        output = result.stdout + result.stderr
        return output.strip() if output.strip() else "Command completed with no output."
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds."
    except Exception as e:
        return f"Error: {e}"


# ── Git ───────────────────────────────────────────────────────────────────────

@mcp.tool
def git_status(repo_path: str) -> str:
    """get the git status of a repository showing changed, staged, and untracked files."""
    result = subprocess.run(["git", "status"], capture_output=True, text=True, cwd=repo_path)
    return result.stdout + result.stderr


@mcp.tool
def git_diff(repo_path: str, file_path: str = "") -> str:
    """get the git diff to see what changed. optionally pass a file_path to diff a specific file."""
    cmd = ["git", "diff"]
    if file_path:
        cmd.append(file_path)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)
    output = result.stdout + result.stderr
    return output.strip() if output.strip() else "No changes detected."


@mcp.tool
def git_commit(repo_path: str, message: str) -> str:
    """stage all changes and create a git commit with the given message."""
    add = subprocess.run(["git", "add", "-A"], capture_output=True, text=True, cwd=repo_path)
    commit = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, cwd=repo_path)
    return add.stdout + add.stderr + commit.stdout + commit.stderr
