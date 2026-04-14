"""
Q5.py - utilities to supply data to the templates.

This file contains a pair of functions for retrieving and manipulating data
that will be supplied to the template for generating the table.
"""
import csv  # for carrying out csv operations
from typing import List, Tuple, Any  # for type hinting


def username():
    return 'urafi3'     # returns username


def data_wrangling(filter_class: str = None) -> Tuple[List[str], List[List[Any]], List[str]]:
    """
    Args:
        - filter_class (str): Optional parameter that specifies the animal class
            to filter the data for.
    """
    with open('data/q5.csv', 'r', encoding='utf-8') as f:    # Open CSV file
        reader = csv.reader(f)
        table = list()
        
        # Read in the header
        for header in reader:
            break
        
        # Read in each row
        for row in reader:
            if len(row) >= 3:  # Ensure row has at least 3 columns
                try:
                    # row[0] = species, row[1] = class, row[2] = count
                    row_data = [row[0], row[1], int(row[2])]
                    table.append(row_data)
                except ValueError:
                    # Skip rows with invalid count values
                    continue
        
        # CORRECT: Get unique CLASS names (row[1]) for dropdown
        dropdown_options = sorted(list(set(row[1] for row in table)))
        
        # Filter, sort, and limit the table
        # Filter the data by the class column (second column)
        if filter_class:
            table = [row for row in table if row[1] == filter_class]
        
        # Order table by the count column (last column) in descending order
        table.sort(key=lambda x: x[2], reverse=True)
        
        # Take only the first 10 rows
        table = table[:10]
    
    return header, table, dropdown_options