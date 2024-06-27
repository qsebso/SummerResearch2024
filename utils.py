from typing import Iterable

def split_seq(seq: list, val) -> list[list]:
    """
    parameters:
        seq: list of anything
        val: something in the list
    returns:
        sublists from seq split everywhere val appears
    """
    result = []
    temp_list = []
    for x in seq:
        if x == val:
            if temp_list:
                result.append(temp_list)
                temp_list = []
        else:
            temp_list.append(x)
    result.append(temp_list)
    return result

def partition_seq(seq: list, percentage: float):
    part_idx = int(len(seq)*percentage)
    return seq[:part_idx], seq[part_idx:]