def clean(input):
    """
    Clean the input string by removing all digit characters.
    Takes:
        input: str
    Returns:
        result: str
    """

    return "".join([i for i in input if not i.isdigit()])


def clean_op(input):
    """
    Clean the input string by removing all non-digit characters.
    Takes:
        input: str
    Returns:
        result: str
    """
    return "".join([i for i in input if i.isdigit()])


def divide_to_list(a, b):
    """
    Divide a to b parts, return a list of length b.
    Takes:
        a: int
        b: int
    Returns:
        result: list of int
    """
    quotient = a // b
    remainder = a % b

    result = []
    for i in range(b):
        increment = 1 if i < remainder else 0
        result.append(quotient + increment)
    return result
