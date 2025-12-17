"""Utility for visualizing directory structures with rich tree output."""

from pathlib import Path
from typing import Optional

from rich.tree import Tree
from rich import print as rprint

def build_tree(
  path: Path,
  tree: Tree,
  max_depth: int = 2,
  depth: int = 0,
  max_items: int = 2,
  max_files: int = 3,
  depth_labels: Optional[dict[int, str]] = None,
  item_count: Optional[dict] = None,
) -> None:
  """Recursively build a rich Tree from a directory structure."""
  if item_count is None:
      item_count = {}

  if depth_labels is None:
      depth_labels = {}

  if depth >= max_depth:
      return

  path = Path(path)

  try:
      entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
  except PermissionError:
      tree.add("[red]Permission denied[/red]")
      return

  files_shown = 0
  for entry in entries:
      if entry.is_dir():
          parent_key = str(entry.parent)
          if parent_key not in item_count:
              item_count[parent_key] = 0

          if item_count[parent_key] >= max_items:
              if item_count[parent_key] == max_items:
                  tree.add("[dim]... and more directories[/dim]")
                  item_count[parent_key] += 1
              continue

          item_count[parent_key] += 1

          label = f" [cyan]({depth_labels[depth]})[/cyan]" if depth in depth_labels else ""
          branch = tree.add(f"📁 {entry.name}{label}")
          build_tree(
              entry, branch, max_depth, depth + 1,
              max_items, max_files, depth_labels, item_count
          )
      else:
          if files_shown >= max_files:
              if files_shown == max_files:
                  remaining = len([e for e in entries if e.is_file()]) - max_files
                  tree.add(f"[dim]... and {remaining} more files[/dim]")
                  files_shown += 1
              continue
          files_shown += 1
          tree.add(f"📄 {entry.name}")


def show_directory_tree(
  root_path: str | Path,
  max_depth: int = 6,
  max_items: int = 2,
  max_files: int = 3,
  depth_labels: Optional[dict[int, str]] = None,
) -> None:
  """Display a directory tree structure using rich."""
  root_path = Path(root_path)
  tree = Tree(f"📁 {root_path}")
  build_tree(
      root_path, tree, max_depth,
      depth=0, max_items=max_items, max_files=max_files,
      depth_labels=depth_labels
  )
  rprint(tree)