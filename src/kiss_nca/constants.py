"""Project-wide constants for The KISS conditional NCA pipeline."""

from __future__ import annotations

from dataclasses import dataclass


PAINTING_IDS: dict[str, int] = {
    "the_kiss": 0,
    "adele_bloch_bauer": 1,
    "tree_of_life": 2,
    "judith": 3,
    "danae": 4,
}

# The checked-in image names differ slightly from the short names used in the
# project brief, so each logical painting name can have one or more aliases.
PAINTING_FILE_ALIASES: dict[str, tuple[str, ...]] = {
    "the_kiss": ("the_kiss.png", "kiss.png"),
    "adele_bloch_bauer": ("adele_bloch_bauer.png", "adele.png"),
    "tree_of_life": ("tree_of_life.png",),
    "judith": ("judith.png",),
    "danae": ("danae.png",),
}


@dataclass(frozen=True)
class ImageSpec:
    """Expected dataset image properties."""

    resolution: int = 64
    channels: tuple[str, ...] = ("RGB", "RGBA")
    extension: str = ".png"
