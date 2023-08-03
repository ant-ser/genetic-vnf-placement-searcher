import re


def split_string_into_matrix(
    input_string: str, row_delimiter: str = "\n", column_delimiter: str = " "
) -> list[list[str]]:
    matrix = [
        row.split(column_delimiter)
        for row in input_string.rstrip(row_delimiter).split(row_delimiter)
    ]
    return matrix


def remove_duplicate_whitespace_characters(string: str) -> str:
    new_string = re.sub("[^\\S\\r\\n]+", " ", string)
    return new_string
