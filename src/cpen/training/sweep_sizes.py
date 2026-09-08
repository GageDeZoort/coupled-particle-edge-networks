"""Parse architecture size sweep strings."""

from __future__ import annotations


def parse_sizes(
    sizes_str: str,
    *,
    default_heads: int = 1,
) -> list[tuple[int, int, int]]:
    """
    Parse ``depth,width;depth,width`` into (depth, width, heads) tuples.

    Optional third field per entry sets heads: ``depth,width,heads``.
    """
    if not sizes_str.strip():
        return []

    sizes: list[tuple[int, int, int]] = []
    for chunk in sizes_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) == 2:
            depth, width = int(parts[0]), int(parts[1])
            heads = default_heads
        elif len(parts) == 3:
            depth, width, heads = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(
                f"Expected 'depth,width' or 'depth,width,heads', got {chunk!r}"
            )
        sizes.append((depth, width, heads))
    return sizes
