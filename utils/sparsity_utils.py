def average_dict(input: dict) -> dict:
    # Returns a dictionary whose values are the average, within keys, of input
    new_dict = {}
    for key in input:
        new_dict[key] = sum(input[key]) / len(input[key])
    return new_dict
