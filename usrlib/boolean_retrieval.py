"""
This module implements functions which are used in boolean retrieval model
"""

from .document import Document
from .invertedindex import InvertedIndex

def AND(list1, list2):
    """
    Perform AND operation on lists
    """

    # does not return empty unless both lists are empty
    # this ensures user gets some results
    if len(list2) == 0:
        return list1
    elif len(list1) == 0:
        return list2
    return [ i for i in list1 if i in list2 ]

def OR(list1, list2):
    """
    Perform OR operation on lists
    """

    return list(set(list1).union(set(list2)))

def parse_query(query, corpus, index):
    """
    This function parses the query and returns relavent files
    """


    qdoc = Document(raw_data=query)
    query_list = list(qdoc.word_freq.keys())
    if len(query_list) < 1:
        return []
    temp = index[query_list[0]]
    i = 1
    while i< len(query_list):
        if query_list[i] == "OR":
            temp = OR(temp, index[query_list[i+1]])
            i += 1
        elif query_list[i] == "AND":
            temp = AND(temp, index[query_list[i+1]])
            i += 1
        else: 
            temp = OR(temp, index[query_list[i]])
        i += 1

    output = []
    for i in temp:
        output.append(i)

    return output

if __name__ == "__main__":
    print("For testing operators only")
    list1 = input("enter list1: ").split()
    list2 = input("enter list2: ").split()
    print(AND(list1, list2))