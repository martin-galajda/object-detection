def build_placeholder_string_from_list(some_list: list):
    placeholder_str = '?, ' * len(some_list)
    placeholder_str = placeholder_str.strip()
    placeholder_str = placeholder_str.strip(',')

    return placeholder_str