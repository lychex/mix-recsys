

def remove(col, df):
    df.drop(columns=col, axis=1, inplace=True)


def duplicate_num_col_count(df, num_col_name):
    """
    Expand a dataframe based on a numerical column
    e.g. if the number of the feature is 2, for this example duplicate twice
    """
    new_df = df.loc[df.index.repeat(df[num_col_name])]
    new_df.drop(columns=num_col_name, axis=1, inplace=True)
    return new_df


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
    """Sort a tuple based on the second value"""
    return(sorted(tup, key=lambda x: x[1]))


def sort_dict(dic, reverse=True):
    """Sort a dictionary based on the value in a descending order"""
    return sorted(dic.items(), key=lambda x: x[1], reverse=reverse)


# Function for sorting tf_idf in descending order
def sort_matrix(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def move_last_col_first(df):
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df_new = df[cols]
    return df_new
