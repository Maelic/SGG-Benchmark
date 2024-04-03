from sgg_benchmark.config.paths_catalog import DatasetCatalog
import os
import json

# we rearrange the VG dataset, sort the relation classes in descending order (the original order is based on relation class names)
# predicate_new_order = [0, 10, 42, 43, 34, 28, 17, 19, 7, 29, 33, 18, 35, 32, 27, 50, 22, 44, 45, 25, 2, 9, 5, 15, 26, 23, 37, 48, 41, 6, 4, 1, 38, 21, 46, 30, 36, 47, 14, 49, 11, 16, 39, 13, 31, 40, 20, 24, 3, 12, 8]
# predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712, 5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352, 663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270, 234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
# predicate_new_order_name = ['__background__', 'on', 'has', 'wearing', 'of', 'in', 'near', 'behind', 'with', 'holding', 'above', 'sitting on', 'wears', 'under', 'riding', 'in front of', 'standing on', 'at', 'carrying', 'attached to', 'walking on', 'over', 'for', 'looking at', 'watching', 'hanging from', 'laying on', 'eating', 'and', 'belonging to', 'parked on', 'using', 'covering', 'between', 'along', 'covered in', 'part of', 'lying on', 'on back of', 'to', 'walking in', 'mounted on', 'across', 'against', 'from', 'growing on', 'painted on', 'playing', 'made of', 'says', 'flying in']

def load_data_statistics(dataset_name, predicate_new_order, predicate_new_order_count, group_size=4):
    data_dir = DatasetCatalog.DATA_DIR
    attrs = DatasetCatalog.DATASETS[dataset_name]
    vg_dict = json.load(open(os.path.join(data_dir, attrs["dict_file"])))
    predicate_new_order_name = []
    for i, p in enumerate(predicate_new_order[1:]):
        p = vg_dict['idx_to_predicate'][str(p)]
        predicate_new_order_name.append(p)

    # generate groups
    predicate_dict = {}
    predicate_new_order.remove(0)
    for idx in range(len(predicate_new_order)):
        predicate_dict[predicate_new_order_name[idx]] =  predicate_new_order[idx]
    j = 1
    for i in predicate_dict.keys():
        predicate_dict[i] = j
        j+=1

    group_list = []
    shunxu_list = []
    clist = []
    slist = []
    counting_list = []
    head_num = predicate_new_order_count[1]
    end_num = int(head_num/group_size)
    idx = 0

    assert(len(predicate_new_order_count[1:]) == len(predicate_new_order_name))

    for name, data in zip(predicate_new_order_name, predicate_new_order_count[1:]):
        idx += 1
        if data >= end_num or end_num < 200:
            clist.append(predicate_dict[name])
            slist.append(idx)
        else:
            counting_list.append([head_num, end_num])
            head_num = data
            end_num = int(data/group_size)
            group_list.append(clist)
            shunxu_list.append(slist)
            clist = [predicate_dict[name]]
            slist = [idx]
    counting_list.append([head_num, end_num])
    group_list.append(clist)
    shunxu_list.append(slist)
    num_count_list = []
    for i in shunxu_list:
        num_count_list.append(len(i))
    
    # print(num_count_list)
    # print(shunxu_list)
    # print(group_list)
    # print(counting_list)

    return group_list, num_count_list


def get_group_splits(Dataset_name, split_name, predicate_new_order, predicate_new_order_count):
    assert split_name in ['divide3', 'divide4', 'divide5', 'average']

    if split_name == 'average':
        incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
        predicate_stage_count = [10, 10, 10, 10, 10]
    else:
        group_size = {'divide3': 5, 'divide4': 4, 'divide5': 3}
        incremental_stage_list, predicate_stage_count = load_data_statistics(Dataset_name, predicate_new_order=predicate_new_order, predicate_new_order_count=predicate_new_order_count, group_size=group_size[split_name])

    assert sum(predicate_stage_count) == 50

    return incremental_stage_list, predicate_stage_count