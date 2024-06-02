import json

'''predicate name and list'''
predicate_lists = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
predicate_dict = {'__background__': 0, 'above': 1, 'across': 2, 'against': 3, 'along': 4, 'and': 5, 'at': 6, 'attached to': 7, 'behind': 8, 'belonging to': 9, 'between': 10, 'carrying': 11, 'covered in': 12, 'covering': 13, 'eating': 14, 'flying in': 15, 'for': 16, 'from': 17, 'growing on': 18, 'hanging from': 19, 'has': 20, 'holding': 21, 'in': 22, 'in front of': 23, 'laying on': 24, 'looking at': 25, 'lying on': 26, 'made of': 27, 'mounted on': 28, 'near': 29, 'of': 30, 'on': 31, 'on back of': 32, 'over': 33, 'painted on': 34, 'parked on': 35, 'part of': 36, 'playing': 37, 'riding': 38, 'says': 39, 'sitting on': 40, 'standing on': 41, 'to': 42, 'under': 43, 'using': 44, 'walking in': 45, 'walking on': 46, 'watching': 47, 'wearing': 48, 'wears': 49, 'with': 50}

indeed_train_sor = [['on', 109355], ['has', 67144], ['wearing', 47326], ['of', 31347],
                    ['in', 21748], ['near', 15300], ['behind', 10011], ['with', 11059], ['holding', 10764],
                    ['above', 6712], ['sitting on', 5086], ['wears', 4810], ['under', 3757], ['riding', 4260], ['in front of', 3167], ['standing on', 2273], ['at', 1829], ['carrying', 1603], ['attached to', 1413], ['walking on', 1225],
                    ['over', 793], ['for', 809], ['looking at', 676], ['watching', 352], ['hanging from', 663], ['laying on', 752], ['eating', 565], ['and', 504], ['belonging to', 644], ['parked on', 601], ['using', 551], ['covering', 460], ['between', 394], ['along', 379], ['covered in', 397], ['part of', 429], ['lying on', 364], ['on back of', 333], ['to', 299], ['walking in', 270], ['mounted on', 234], ['across', 171], ['against', 208], ['from', 163], ['growing on', 157], ['painted on', 151], ['playing', 71], ['made of', 114], ['says', 44], ['flying in', 4]]

def generate_groups_by_n_times(times=4):
    '''
    Get your own groups!
    For every element in group, the maximal amount of training instances will be no more than x times of the minimal
    '''
    group_list = []
    shunxu_list = []
    clist = []
    slist = []
    counting_list = []
    head_num = indeed_train_sor[0][1]
    end_num = int(head_num/times)
    idx = 0
    for name, data in indeed_train_sor:
        idx += 1
        if data >= end_num or end_num < 200:
            clist.append(predicate_dict[name])
            slist.append(idx)
        else:
            counting_list.append([head_num, end_num])
            head_num = data
            end_num = int(data/times)
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
    print(num_count_list)
    print(shunxu_list)
    print(group_list)
    print(counting_list)
    return group_list

def generate_groups_by_n_times_GQA(ID_info, times=3):
    with open(ID_info, 'r') as f:
        data_json = json.load(f)
    rel_count = data_json['rel_count']
    id_to_predicate = data_json['rel_name_to_id']
    id_to_predicate.pop('__background__')
    rel_count.pop('__background__')
    rel_count = sorted(rel_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    group_list = []
    shunxu_list = []
    clist = []
    slist = []
    counting_list = []
    head_num = rel_count[0][1]
    end_num = int(head_num/times)
    idx = 0
    for rel_data in rel_count:
        idx += 1
        name, count = rel_data
        if count >= end_num or end_num < 200:
            clist.append(id_to_predicate[name])
            slist.append(idx)
        else:
            counting_list.append([head_num, end_num])
            head_num = count
            end_num = int(count/times)
            group_list.append(clist)
            shunxu_list.append(slist)
            clist = [id_to_predicate[name]]
            slist = [idx]
    counting_list.append([head_num, end_num])
    group_list.append(clist)
    shunxu_list.append(slist)
    num_count_list = []
    for i in shunxu_list:
        num_count_list.append(len(i))
    print(num_count_list)
    print(shunxu_list)
    print(group_list)
    print(counting_list)
    return group_list

if __name__ == '__main__':
    ID_info = 'datasets/gqa/GQA_200_ID_Info.json'
    generate_groups_by_n_times(times=4)
    print('\n\n')
    generate_groups_by_n_times_GQA(ID_info, times=4)