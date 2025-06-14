import argparse
from typing import List, Union, Optional


def parse_layer_index(value: str) -> Union[List[int], int, str]:
    """
    Parse a layer index string into a list of integers, a single integer, or a special value.

    Args:
        value (str): The layer index string (e.g., "1,2,3", "-1", "all").

    Returns:
        Union[List[int], int, str]: Parsed layer indices or special values.

    Raises:
        argparse.ArgumentTypeError: If the input format is invalid.
    """
    value = value.strip()

    # Case 1: special values
    if value == "-1":
        return -1
    if value.lower() == "all":
        return "all"

    # Case 2: list of integers like "1,2,3"
    try:
        int_list = [int(v.strip()) for v in value.split(",")]
        return int_list
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated integers, '-1', or 'all'"
        )


def parse_question_instruction(value: Optional[str]) -> Optional[str]:
    """
    Parses a question instruction value and returns a standardized string or None.

    Acceptable values:
        - 'singular' -> returns 'singular'
        - 'plural'   -> returns 'plural'
        - 'none'     -> returns None
        - empty string -> returns None

    Raises:
        argparse.ArgumentTypeError: If the input is not one of the accepted values.
    """
    if value is None:
        return None

    normalized = value.strip().lower()

    if normalized in {"", "none"}:
        return None
    elif normalized in {"singular", "plural"}:
        return normalized
    else:
        raise argparse.ArgumentTypeError(
            "Expected one of: 'singular', 'plural', 'none', or empty."
        )


# create parser for tokens_mode which can only be 'last' or 'mean'
def parse_tokens_mode(value: str) -> str:
    """
    Parses the tokens_mode argument to ensure it is either 'last' or 'mean'.
    """
    value = value.strip().lower()
    if value in {"last", "mean"}:
        return value
    else:
        raise argparse.ArgumentTypeError("Expected 'last' or 'mean' for tokens_mode.")
