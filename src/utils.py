def strip_split(s: str, sep=None, item_type=None):
    """Strip a string of whitespace and split it apart given a separator character."""
    s = s.strip()
    s = s.split(sep)
    if item_type is int:
        return [int(x) for x in s]
    elif item_type is float:
        return [float(x) for x in s]
    elif item_type is None:
        return s
    else:
        raise ValueError(f'[{item_type}] Invalid item type. Choose None, int, or float')