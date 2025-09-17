# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ffcount",
# ]
# ///

# ruff: noqa: T201
# ruff: noqa: PTH116
# ruff: noqa: PTH118
# ruff: noqa: PTH108
# ruff: noqa: PTH106
from __future__ import annotations

import argparse
import asyncio
import gc
import os
import pathlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ffcount import ffcount

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator


@dataclass(slots=True)
class StackFrame:
    path: str
    iterator: Iterator[os.DirEntry] | None
    kept: bool


logged_empty_stats = False


def copy_folder_tree(original_folder: pathlib.Path, temp_dir: pathlib.Path, non_empty: str = "t") -> None:
    """
    Recreates the entire directory tree structure of `original_folder`
    in a new temporary folder, but with all files replaced by empty files.

    if a file name has the string `non_empty` in it, random content will be written to it
    """

    # print(f"Creating duplicate of directory: {original_folder}")
    global logged_empty_stats
    start = time.perf_counter_ns()

    empty_files, non_empty_files = 0, 0
    for root, sub_folders, files in os.walk(original_folder):
        base_dir = pathlib.Path(root)
        temp_base = temp_dir / base_dir.relative_to(original_folder)

        for folder in sub_folders:
            temp_folder = temp_base / folder
            temp_folder.mkdir(parents=True, exist_ok=True)

        for name in files:
            temp_file = temp_base / name
            if non_empty in name:
                temp_file.write_bytes(b"temp file")
                non_empty_files += 1
            else:
                temp_file.touch()
                empty_files += 1

    end = time.perf_counter_ns()
    creation_time = (end - start) / 1e9
    if logged_empty_stats:
        return
    print(
        f"Created temporary directory duplicate: {temp_dir}. "
        f"(Took: {(creation_time):.3f}s) \n- {empty_files = :,} \n- {non_empty_files = :,}"
    )
    logged_empty_stats = True


def purge_dir_tree_new(dirname: Path) -> None:
    """Purges empty files and directories efficiently."""
    if not dirname.is_dir():
        return

    def get_size(entry: os.DirEntry) -> int | None:
        try:
            if entry.is_file(follow_symlinks=False):
                return entry.stat(follow_symlinks=False).st_size
        except (OSError, PermissionError):
            pass
        return None

    stack: list[StackFrame] = [StackFrame(str(dirname), None, False)]

    while stack:
        frame = stack[-1]

        if frame.iterator is None:
            try:
                frame.iterator = os.scandir(frame.path)
            except (OSError, PermissionError):
                frame.iterator = iter(())
                frame.kept = True
            continue

        try:
            entry = next(frame.iterator)
        except StopIteration:
            try:
                frame.iterator.close()
            except Exception:
                pass

            # Remove empty directories
            if not frame.kept:
                try:
                    os.rmdir(frame.path)
                except OSError:
                    pass

            # Pop the frame and propagate kept status to parent
            stack.pop()
            if stack and frame.kept:
                parent = stack[-1]
                if not parent.kept:
                    parent.kept = True
            continue

        try:
            # Remove empty files
            if entry.is_file(follow_symlinks=False):
                if get_size(entry) == 0:
                    try:
                        os.unlink(entry.path)
                    except OSError:
                        frame.kept = True
                else:
                    frame.kept = True
            elif entry.is_dir(follow_symlinks=False):
                stack.append(StackFrame(entry.path, None, False))
            else:
                frame.kept = True
        except (OSError, PermissionError):
            frame.kept = True


def walk_tree(dirname: Path | str) -> Generator[tuple[str, list[os.DirEntry[str]], list[os.DirEntry[str]]]]:
    """like `os.walk`, but always yields bottom to top and yields `os.DirEntry`s instead of strings"""
    dirs: list[os.DirEntry[str]] = []
    files: list[os.DirEntry[str]] = []

    try:
        for entry in os.scandir(dirname):
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            if is_dir:
                dirs.append(entry)
            else:
                files.append(entry)

    except (OSError, PermissionError):
        pass

    for entry in dirs:
        yield from walk_tree(entry.path)

    yield os.fspath(dirname), dirs, files


async def purge_dir_async(dirname: Path) -> None:
    """Purges empty files and directories efficiently."""
    if not dirname.is_dir():
        return

    def get_size(path: pathlib.Path | str) -> int | None:
        try:
            return os.stat(path, follow_symlinks=False).st_size
        except (OSError, PermissionError):
            return

    def try_delete_file(path: pathlib.Path | str) -> None:
        if get_size(path) == 0:
            try:
                os.unlink(path)
            except OSError:
                return

    def try_delete_dir(path: pathlib.Path | str) -> None:
        try:
            os.rmdir(path)
        except OSError:
            return

    # Fist pass to delete all empty files async
    dirs: list[str] = []
    async with asyncio.TaskGroup() as tg:
        for root, sub_folders, files in os.walk(dirname):
            if not sub_folders and not files:
                # We found an already empty folder. Schedule it for deletion
                tg.create_task(asyncio.to_thread(try_delete_dir, root))
                continue

            for file_name in files:
                file_path = os.path.join(root, file_name)
                tg.create_task(asyncio.to_thread(try_delete_file, file_path))
            dirs.append(root)

    try:
        while dir := dirs.pop():
            try_delete_dir(dir)
    except IndexError:
        return


def purge_dir_tree_original(dirname: Path) -> None:
    """Purges empty files and directories efficiently."""
    if not dirname.is_dir():
        return

    def get_size(path: Path):
        try:
            return path.stat().st_size
        except (OSError, ValueError):
            return

    # Use os.walk() to remove empty files and directories in a single pass
    for dirpath, _dirnames, filenames in os.walk(dirname, topdown=False):
        dir_path = Path(dirpath)

        # Remove empty files
        has_non_empty_files = False
        for file_name in filenames:
            file_path = dir_path / file_name
            if get_size(file_path) == 0:
                file_path.unlink()
            else:
                has_non_empty_files = True

        # Remove empty directories
        if not has_non_empty_files:
            try:
                dir_path.rmdir()
            except OSError:
                continue


def purge_dir_tree_original_v2(dirname: Path) -> None:
    """Purges empty files and directories efficiently."""
    if not dirname.is_dir():
        return

    def get_size(path: Path):
        try:
            return path.stat().st_size
        except (OSError, ValueError):
            return

    # Use os.walk() to remove empty files and directories in a single pass
    for dirpath, dirnames, filenames in os.walk(dirname, topdown=False):
        dir_path = Path(dirpath)

        # Remove empty files
        has_non_empty_files = False
        for file_name in filenames:
            file_path = dir_path / file_name
            if get_size(file_path) == 0:
                file_path.unlink()
            else:
                has_non_empty_files = True

        # Remove empty directories
        if not dirnames and not has_non_empty_files:
            try:
                dir_path.rmdir()
            except OSError:
                continue


def purge_dir_tree_original_v3(dirname: Path) -> None:
    if not dirname.is_dir():
        return

    def get_size(path: os.DirEntry):
        try:
            return path.stat(follow_symlinks=False).st_size
        except (OSError, ValueError):
            return

    # Use os.walk() to remove empty files and directories in a single pass
    for dirpath, dirnames, filenames in walk_tree(dirname):
        # Remove empty files
        has_non_empty_files = False
        for entry in filenames:
            if get_size(entry) == 0:
                os.unlink(entry)
            else:
                has_non_empty_files = True

        # Remove empty directories
        if not dirnames and not has_non_empty_files:
            try:
                os.rmdir(dirpath)
            except OSError:
                continue


def purge_dir_tree_original_v4(dirname: Path) -> None:
    if not dirname.is_dir():
        return

    def get_size(path: os.DirEntry) -> int | None:
        try:
            return path.stat(follow_symlinks=False).st_size
        except (OSError, ValueError):
            return

    for dirpath, _dirnames, filenames in walk_tree(dirname):
        has_non_empty_files = False
        for entry in filenames:
            if get_size(entry) == 0:
                os.unlink(entry)
            else:
                has_non_empty_files = True

        if has_non_empty_files:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            continue


def time_once(func: Callable[..., None], *args) -> float:
    """
    Time a single call, returning elapsed nanoseconds.
    GC is disabled during the timed section for stability.
    """
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    start = time.perf_counter_ns()
    func(*args)
    elapsed = time.perf_counter_ns() - start
    if gc_enabled:
        gc.enable()
    return elapsed


def run_benchmark(original_folder: pathlib.Path, repeats: int) -> None:
    original_folder = validate_folder(original_folder)

    f"Created temporary directory duplicate for {original_folder}.\n"

    files, folders = ffcount(original_folder)
    print(f"{original_folder.name}: {files:,} files in {folders:,} folders")

    with tempfile.TemporaryDirectory(prefix="cdl_test_") as tmp:
        temp_dir = pathlib.Path(tmp)

        def setup() -> None:
            copy_folder_tree(original_folder, temp_dir)

        def run_test(label: str, func: Callable[[pathlib.Path], None]) -> None:
            print("-" * 30)
            print(f"\n{label}")

            def run(n: int) -> float:
                setup()
                result = time_once(func, temp_dir)
                return result

            runs = [run(n) for n in range(repeats)]
            temp_files, temp_folders = ffcount(temp_dir)
            deleted = files - temp_files, folders - temp_folders
            print_stats(runs, *deleted)

        def run_async(path: pathlib.Path) -> None:
            return asyncio.run(purge_dir_async(path))

        run_test("== Old Method ==", purge_dir_tree_original)
        run_test("== Old Method v2 ==", purge_dir_tree_original_v2)
        run_test("== Old Method v3 ==", purge_dir_tree_original_v3)
        run_test("== Old Method v4 ==", purge_dir_tree_original_v4)
        run_test("== New Method ==", purge_dir_tree_new)
        run_test("== Async Method ==", run_async)
        run_test("== Async Method w threads precreated ==", run_async)


def validate_folder(original_folder: pathlib.Path) -> Path:
    original_folder = original_folder.resolve()
    if not original_folder.exists():
        raise FileNotFoundError(f"Original folder not found: {original_folder}")
    if not original_folder.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {original_folder}")
    assert original_folder.is_absolute()
    return original_folder


def print_stats(runs: list[float], deleted_files: int, deleted_folders: int) -> None:
    best = min(runs) / 1e9
    worst = max(runs) / 1e9
    mean = (sum(runs) / len(runs)) / 1e9
    runs_str = ", ".join(f"{r / 1e9:.3f}" for r in runs)
    print(f"\n- Min: {best:.3f}s")
    print(f"- Max: {worst:.3f}s")
    print(f"- Mean: {mean:.3f}s")
    print(f"- Runs: {runs_str}s")
    print(f"- Deleted files: {deleted_files:,}")
    print(f"- Deleted folders: {deleted_folders:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test speeds for removing empty files and folders")
    parser.add_argument("folder", type=Path, help="folder to recreate")
    parser.add_argument("-n", type=int, default=5, help="Number of runs per directory (default: 1)")
    args = parser.parse_args()

    run_benchmark(args.folder, args.n)


if __name__ == "__main__":
    main()
