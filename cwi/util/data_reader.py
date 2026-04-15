# -*- coding: utf-8 -*-

"""Reader functions to help with data preparation."""

import pandas as pd


def create_raw_instances(dataframe, sent_col, start_col, end_col, label_col=""):
    """Create instances of the form of (sentence, start_index, end_index).

    Args:
        dataframe(pd.DataFrame): pandas dataframe containing the instances
        sent_col (str): name of the column containing the entire sentence
        start_col (str): name of the column containing the start index of the
            target span
        end_col (str): name of the column containing the end index of the
            target span
        label_col (str): name of the column containing the label of the target,
            if not provided no labels will be returned

    Returns:
        list[tuple]: instances as a list of the aforementioned tuple format
        pd.Series: labels (optional)
    """
    # Get instance tuples from dataframe
    instances = list(
        dataframe[[sent_col, start_col, end_col]].itertuples(index=False,
                                                             name=None)
    )
    # Return labels as well if label_col has been provided
    if label_col:
        labels = dataframe[label_col]
        return instances, labels
    return instances


def read_sharedtask(file_path):
    """Read a dataframe with shared task data and generate raw instances."""
    # Column names of dataframe
    col_names = [
        "ID",
        "sentence",
        "start_idx_target",
        "end_idx_target",
        "target_text",
        "full_native_annotators",
        "full_nonnative_annotators",
        "native_positives",
        "nonnative_positives",
        "gold_binary",
        "gold_probabilistic"
    ]
    df = pd.read_csv(file_path, sep="\t", names=col_names)
    # Set columns to be used
    sent_col = "sentence"
    start_col = "start_idx_target"
    end_col = "end_idx_target"
    label_col = "gold_binary"
    # Generate and return raw instances from dataframe
    return create_raw_instances(df, sent_col, start_col, end_col, label_col)


def read_mdr(file_path):
    """Read a dataframe with shared task data and generate raw instances."""
    df = pd.read_csv(file_path, encoding="utf-8")
    # Set columns to be used
    sent_col = "sent"
    start_col = "start"
    end_col = "end"
    label_col = "label"
    # Generate and return raw instances from dataframe
    return create_raw_instances(df, sent_col, start_col, end_col, label_col)

