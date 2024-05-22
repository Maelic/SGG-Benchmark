from PIL import Image, ImageDraw, ImageFont
import random
import os
import json
import numpy as np
import altair as alt

USE_BOX_SIZE = 1024
FONT_SIZE=20

# download the font file from https://www.fontpalace.com/font-details/Arial/ in cache folder
font_file_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"



def draw_single_label(pic, box, label, color=(255,0,255,128)):
    draw = ImageDraw.Draw(pic)
    # change font size and type
    font = ImageFont.truetype(font_file_path, FONT_SIZE)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    p1 = (x1, y1)
    p2 = (x2, y2)
    bb_center = (int(p1[0] + p2[0])/2, int(p1[1] + p2[1])/2)

    bbox = draw.textbbox(bb_center, label, font=font)
    draw.rectangle(bbox, fill=color)
    draw.text(bb_center, label, fill="black", font=font)
    return bb_center

def draw_single_box(draw, box, label, color='red'):
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
    draw.text((x1, y1), label)

def draw_boxes(pic, boxes, obj_labels):
    draw = ImageDraw.Draw(pic)
    font = ImageFont.truetype(font_file_path, 11)
    assert(len(boxes) == len(obj_labels))
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        draw.rectangle(((x1, y1), (x2, y2)), outline='red')
        
        text_width, text_height = draw.textsize(obj_labels[i], font=font)
        draw.rectangle(((x1, y1), (x1 + text_width, y1 + text_height)), fill='red')
        
        draw.text((x1, y1), obj_labels[i], font=font)

def get_images_rel(vg_sgg, vg_sgg_dicts, img_idx):
    idx_to_label = vg_sgg_dicts['idx_to_label']
    idx_to_predicate = vg_sgg_dicts['idx_to_predicate']
    ith_s = vg_sgg['img_to_first_rel'][img_idx]
    ith_e = vg_sgg['img_to_last_rel'][img_idx]
    rel_idx = random.randint(ith_s, ith_e) 
    objs = vg_sgg['relationships'][rel_idx]
    label = vg_sgg['labels'][objs[0]]
    pred = vg_sgg['predicates'][rel_idx]

    res = idx_to_label[str(int(vg_sgg['labels'][objs[0]]))], \
        idx_to_predicate[str(int(pred))], \
        idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]
    return res

def get_all_images_rel(vg_sgg, vg_sgg_dicts, img_idx):
    idx_to_label = vg_sgg_dicts['idx_to_label']
    idx_to_predicate = vg_sgg_dicts['idx_to_predicate']
    ith_s = vg_sgg['img_to_first_rel'][img_idx]
    ith_e = vg_sgg['img_to_last_rel'][img_idx]
    res = []
    for rel_idx in range(ith_s, ith_e): 
        objs = vg_sgg['relationships'][rel_idx]
        label = vg_sgg['labels'][objs[0]]
        pred = vg_sgg['predicates'][rel_idx]

        res.append([idx_to_label[str(int(vg_sgg['labels'][objs[0]]))], \
            idx_to_predicate[str(int(pred))], \
            idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]])
    return res

def get_random_imgs_by_pred(num_images, vg_sgg, vg_sgg_dicts, pred):
    pred_idx = int(vg_sgg_dicts['predicate_to_idx'][pred])-1
    vg_rel_dict = json.load(open("./VG-rel-dict.json", 'r'))
    imgs_id = []
    rels_id = []
    i=0
    while i < num_images:
        rd = random.choice(vg_rel_dict[str(pred_idx)])
        if rd[0] not in imgs_id:
            imgs_id.append(rd[0])
            rels_id.append(rd[1])
            i=i+1
            
    return imgs_id, rels_id

def get_random_imgs_by_triplet(num_images, vg_sgg, vg_sgg_dicts, triplet):
    subject_idx = int(vg_sgg_dicts['label_to_idx'][triplet[0]])-1
    object_idx = int(vg_sgg_dicts['label_to_idx'][triplet[2]])-1
    pred_idx = int(vg_sgg_dicts['predicate_to_idx'][triplet[1]])-1

    r = list(range(len(vg_sgg['relationships'])))
    random.shuffle(r)
    found = 0
    output_rel  = []

    for i in r:

        pred = vg_sgg['predicates'][i][0]

        sub = vg_sgg['relationships'][i][0]
        obj = vg_sgg['relationships'][i][1]

        sub_num = vg_sgg['labels'][sub][0]
        obj_num = vg_sgg['labels'][obj][0]

        if subject_idx == sub_num and object_idx == obj_num and pred_idx == pred:
            print(i)
            output_rel.append(i)
            found += 1
            if found == num_images:
                break

    imgs_id = []
    for rel in output_rel:
        for i in range(len(vg_sgg['img_to_first_rel'])):
            if vg_sgg['img_to_first_rel'][i] <= rel and vg_sgg['img_to_last_rel'][i] >= rel:
                imgs_id.append(i)
                break
    return imgs_id, rels_id

def show_rel_on_image(image_data, vg_sgg, vg_sgg_dicts, img_idx, rel_idx, vg_img_path):
    idx_to_label = vg_sgg_dicts['idx_to_label']
    idx_to_predicate = vg_sgg_dicts['idx_to_predicate']

    objs = vg_sgg['relationships'][rel_idx]
    pred = vg_sgg['predicates'][rel_idx]

    height, width = image_data[img_idx]['height'], image_data[img_idx]['width']

    box1 = vg_sgg['boxes_1024'][objs[0]]
    box2 = vg_sgg['boxes_1024'][objs[1]]

    box1[:2] = box1[:2] - box1[2:] / 2
    box1[2:] = box1[:2] + box1[2:]
    box1 = box1.astype(float) / USE_BOX_SIZE * max(height, width)

    box2[:2] = box2[:2] - box2[2:] / 2
    box2[2:] = box2[:2] + box2[2:]
    box2 = box2.astype(float) / USE_BOX_SIZE * max(height, width)
    obj_labels = [idx_to_label[str(int(vg_sgg['labels'][objs[0]]))], idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]]
    pic = draw_boxes(image_data[img_idx]['image_id'], [box1, box2], obj_labels, vg_img_path)

    label = [idx_to_label[str(int(vg_sgg['labels'][objs[0]]))], \
            idx_to_predicate[str(int(pred))], \
            idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]]
    
    print('Label: {}'.format(label))
    return display(pic)

def draw_rel(pic, c1,c2, label, color=(0,255,255,128)):
    draw = ImageDraw.Draw(pic)
    font = ImageFont.truetype(font_file_path, FONT_SIZE)
    draw.line((c1, c2), width=3, fill=color)
    bb_center = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
    bbox = draw.textbbox(bb_center, label, font=font)
    draw.rectangle(bbox, fill=color)
    draw.text(bb_center, label, fill="black", font=font)

    return bb_center

def show_all_boxes_on_image(image_data, vg_sgg, vg_sgg_dicts, img_idx, vg_img_path):
    idx_to_label = vg_sgg_dicts['idx_to_label']
    idx_to_predicate = vg_sgg_dicts['idx_to_predicate']
    labels=[]
    pic = Image.open(os.path.join(vg_img_path, '{}.jpg'.format(image_data[img_idx]['image_id'])))
    print("Image ID: {}".format(image_data[img_idx]['image_id']))
    ith_s = vg_sgg['img_to_first_rel'][img_idx]
    ith_e = vg_sgg['img_to_last_rel'][img_idx]
    res = []
    img_to_first_box = vg_sgg['img_to_first_box'][img_idx]
    for rel_idx in range(ith_s, ith_e): 
        objs = vg_sgg['relationships'][rel_idx]
        pred = vg_sgg['predicates'][rel_idx]

        res.append([idx_to_label[str(int(vg_sgg['labels'][objs[0]]))], \
            idx_to_predicate[str(int(pred))], \
            idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]])


        height, width = image_data[img_idx]['height'], image_data[img_idx]['width']

        box1 = vg_sgg['boxes_1024'][objs[0]]
        box2 = vg_sgg['boxes_1024'][objs[1]]
        label1 = str(objs[0]-img_to_first_box)+'_'+idx_to_label[str(int(vg_sgg['labels'][objs[0]]))]
        label2 = str(objs[1]-img_to_first_box)+'_'+idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]
        pred_label =idx_to_predicate[str(int(pred))]

        box1[:2] = box1[:2] - box1[2:] / 2
        box1[2:] = box1[:2] + box1[2:]
        box1 = box1.astype(float) / USE_BOX_SIZE * max(height, width)

        box2[:2] = box2[:2] - box2[2:] / 2
        box2[2:] = box2[:2] + box2[2:]
        box2 = box2.astype(float) / USE_BOX_SIZE * max(height, width)

        draw_boxes(pic, [box1, box2], [label1,label2])

        labels.append([label1, pred_label, label2])

    print('Relations: \n')
    string=""
    for i in labels:
        string+=i[0]+' '+i[1]+' '+i[2]+', '
    print(string)
    return display(pic)


def show_all_rel_on_image(image_data, vg_sgg, vg_sgg_dicts, img_idx, vg_img_path):
    idx_to_label = vg_sgg_dicts['idx_to_label']
    idx_to_predicate = vg_sgg_dicts['idx_to_predicate']
    labels=[]
    pic = Image.open(os.path.join(vg_img_path, '{}.jpg'.format(image_data[img_idx]['image_id'])))
    print("Image ID: {}".format(image_data[img_idx]['image_id']))
    ith_s = vg_sgg['img_to_first_rel'][img_idx]
    ith_e = vg_sgg['img_to_last_rel'][img_idx]
    res = []
    for rel_idx in range(ith_s, ith_e): 
        objs = vg_sgg['relationships'][rel_idx]
        label = vg_sgg['labels'][objs[0]]
        pred = vg_sgg['predicates'][rel_idx]

        res.append([idx_to_label[str(int(vg_sgg['labels'][objs[0]]))], \
            idx_to_predicate[str(int(pred))], \
            idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]])


        height, width = image_data[img_idx]['height'], image_data[img_idx]['width']

        box1 = vg_sgg['boxes_1024'][objs[0]]
        box2 = vg_sgg['boxes_1024'][objs[1]]
        label1=idx_to_label[str(int(vg_sgg['labels'][objs[0]]))]
        label2=idx_to_label[str(int(vg_sgg['labels'][objs[1]]))]
        pred_label =idx_to_predicate[str(int(pred))]

        box1[:2] = box1[:2] - box1[2:] / 2
        box1[2:] = box1[:2] + box1[2:]
        box1 = box1.astype(float) / USE_BOX_SIZE * max(height, width)

        box2[:2] = box2[:2] - box2[2:] / 2
        box2[2:] = box2[:2] + box2[2:]
        box2 = box2.astype(float) / USE_BOX_SIZE * max(height, width)

        c1 = draw_single_label(pic, box1, label1)
        c2 = draw_single_label(pic, box2, label2)
        draw_rel(pic, c1, c2, pred_label)
        # draw_boxes(pic, [box1, box2], [label1,label2])

        labels.append([label1, pred_label, label2])

    print('Relations: \n')
    string=""
    for i in labels:
        string+=i[0]+' '+i[1]+' '+i[2]+', '
    print(string)
    return display(pic)

def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab
    
def write_tsv(tsv_path, data):
    with open(tsv_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['node1', 'relation', 'node2'])
        for row in data:
            writer.writerow(row)

def bar_chart(data, x_column, y_column, title="", width=800, height=400):
    """Construct a simple bar chart with two properties"""
    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y(y_column, sort='-x'),
        x=x_column
    ).properties(
        title=title,
        width=width,
        height=height
    )

    return (bars)

def bar_chart_text(data, x_column, y_column, title="", width=800, height=400):
    """Construct a simple bar chart with two properties"""
    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y(y_column, sort='-x'),
        x=x_column
    ).properties(
        title=title,
        width=width,
        height=height
    )
    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text=x_column
    )

    return (bars + text)
    
def bar_chart_color(data, x_column, y_column, title="", width=800, height=400):
    """Construct a simple bar chart with two properties"""
    bars = alt.Chart(data).mark_bar().encode(
        color=alt.Color('type', scale=None, legend=None),
        y=alt.Y(y_column,sort=alt.EncodingSortField(field='count', order='descending', op='sum')),
        x=x_column
    ).properties(
        title=title,
        width=width,
        height=height
    )

    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text=x_column
    )

    return (bars + text)