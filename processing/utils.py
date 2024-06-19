def split_seq(seq, val):
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

def partition_seq(seq, percentage):
    part_idx = int(len(seq)*percentage)
    return seq[:part_idx], seq[part_idx:]