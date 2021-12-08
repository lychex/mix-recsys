def remove(col, df):
    df.drop(columns=col, axis=1, inplace=True)


def flatten_list(nested_list):
    """Converts a nested list to a flat list"""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def sort_tuple(tup):
    return(sorted(tup, key=lambda x: x[1]))


def sort_dict(dic, reverse=True):
    return sorted(dic.items(), key=lambda x: x[1], reverse=reverse)


# Function for sorting tf_idf in descending order
def sort_matrix(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
