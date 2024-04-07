import torch
import torch.nn.functional as F
import numpy as np

import time
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def testing_topk_km_imbalance_matsize_np2(N=1000):
    #n, m = np.random.choice(N, 2)
    #m = n * 10
    n = 300
    m = 100*100 - 100 - 10
    cost = np.random.randn(n, m)
    a = time.time()
    row_ind, col_ind = linear_sum_assignment(cost)
    b1 = time.time()-a
    
    tc = cost
    
    vis_table = np.zeros(cost.shape[-1], dtype=np.bool)
    
    a = time.time()
    l, r = 2, max(n//m, m//n)
    s_class = np.argpartition(tc, r)
    s_class = s_class[:, :r]
    #ans = np.unique(s_class.reshape(-1))
    vis_table[s_class.reshape(-1)] = True
    ans = np.where(vis_table)[0]
    if m > n:
        while l <= r:
            vis_table[:] = 0
            mid = (l + r) // 2
            #s_score, s_class = torch.topk(tc, k=mid, largest=False, dim=-1)
            s_class = np.argpartition(tc, mid)
            s_class = s_class[:, :mid]
            #unique_ans = np.unique(s_class.reshape(-1))
            vis_table[s_class.reshape(-1)] = True
            if vis_table.sum() >= min(n, m):
                r = mid - 1
                ans = vis_table
            else:
                l = mid + 1
    
    ans = np.where(vis_table)[0]
    row_ind2, col_ind2 = linear_sum_assignment(cost[:, ans])
    b2 = time.time()-a
    
    #print(cost[row_ind, col_ind])
    #print(cost[:, ans][row_ind2, col_ind2])
    print((cost[row_ind, col_ind] == cost[:, ans][row_ind2, col_ind2]).any())
    print(b1, b2, len(ans), n, m)




def testing_topk_km_imbalance_matsize_np(N=1000):
    #n, m = np.random.choice(N, 2)
    #m = n * 10
    n = 300
    m = 100*100 - 100 - 10
    cost = np.random.randn(n, m)
    a = time.time()
    row_ind, col_ind = linear_sum_assignment(cost)
    b1 = time.time()-a
    
    tc = cost
    
    a = time.time()
    l, r = 2, max(n//m, m//n)
    s_class = np.argpartition(tc, r)
    s_class = s_class[:, :r]
    ans = np.unique(s_class.reshape(-1))
    if m > n:
        while l <= r:
            mid = (l + r) // 2
            #s_score, s_class = torch.topk(tc, k=mid, largest=False, dim=-1)
            s_class = np.argpartition(tc, mid)
            s_class = s_class[:, :mid]
            unique_ans = np.unique(s_class.reshape(-1))
            print(len(unique_ans), mid)
            if len(unique_ans) >= min(n, m):
                r = mid - 1
                ans = unique_ans
            else:
                l = mid + 1
            
    row_ind2, col_ind2 = linear_sum_assignment(cost[:, ans])
    b2 = time.time()-a
    
    #print(cost[row_ind, col_ind])
    #print(cost[:, ans][row_ind2, col_ind2])
    print((cost[row_ind, col_ind] == cost[:, ans][row_ind2, col_ind2]).any())
    print(b1, b2, len(ans), n, m)



def testing_topk_km_imbalance_matsize(N=1000):
    #n, m = np.random.choice(N, 2)
    #m = n * 10
    n = 300
    m = 100*100 - 100 - 10
    cost = np.random.randn(n, m)
    a = time.time()
    row_ind, col_ind = linear_sum_assignment(cost)
    b1 = time.time()-a
    
    tc = torch.tensor(cost).cuda(0)
    
    a = time.time()
    l, r = 1, min(n, m)
    r = 10
    s_score, s_class = torch.topk(tc, k=min(n, m), largest=False, dim=-1)
    ans, _ = torch.unique(s_class.view(-1), return_inverse=True)
    if m > n:
        while l <= r:
            mid = (l + r) // 2
            s_score, s_class = torch.topk(tc, k=mid, largest=False, dim=-1) # 
            unique_ans, _ = torch.unique(s_class.view(-1), return_inverse=True)
            if len(unique_ans) >= min(n, m):
                r = mid - 1
                ans = unique_ans
            else:
                l = mid + 1
            
    ans = ans.data.cpu().numpy()
    
    row_ind2, col_ind2 = linear_sum_assignment(cost[:, ans])
    b2 = time.time()-a
    
    print(cost[row_ind, col_ind])
    print(cost[:, ans][row_ind2, col_ind2])
    print((cost[row_ind, col_ind] == cost[:, ans][row_ind2, col_ind2]).any())
    print(b1, b2, len(ans), n, m)
    

def exam_cost_mat(cost, top_k=10):
    top_k = min(top_k, cost.shape[-1])
    s_score, s_class = torch.topk(cost, k=top_k, largest=False, dim=-1) # 
    s_score, sorting_idx = torch.sort(s_score.view(-1), dim=0, descending=False)
    s_class = s_class.view(-1)
    s_class = s_class[sorting_idx]
    s_class = s_class.data.cpu().numpy()
    _, unique_indices = np.unique(s_class, return_index=True)
    unique_indices = np.sort(unique_indices)
    s_class = s_class[unique_indices]
    print(s_score[:top_k], s_score[-max(top_k//2, 1):], len(s_class))

def exam_svd(mmodel):
    params = mmodel.named_parameters()
    for name, param in params:
        if len(param.data.shape) != 2:
            continue
        u,s,v = torch.svd(param.data)
        print(name, param.data.shape, len(s), s.min(), s.max())
    
    assert False

def debug_tsne(feat, idx=0, fname='obj'):
    print(feat.shape)
    if feat.shape[-1] == 2048: d = feat.shape[-1] // 2
    else: d = feat.shape[-1]
    if feat.shape[-1] == 512: d = feat.shape[-1] // 2
    else: d = feat.shape[-1]
    feat = feat.view(-1, d)
    feat = feat.data.cpu().numpy()
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(feat)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],label="t-SNE")
    plt.legend()
    plt.savefig('{}_tsne{}.png'.format(fname, idx))
    plt.show()

def debug_rect(bboxes, images_whwh, idx=0, ori_images=None, fname=''):
    rgb_mean = np.array([102.9801, 115.9465, 122.7717]).reshape(3,1,1)
    
    print(bboxes)
    bboxes = bboxes.clone().detach()
    images_whwh = images_whwh.clone().detach()
    w,h = images_whwh[0, :2]
    bboxes = bboxes.view(-1, 4)
    
    bboxes = bboxes[:2]
    #bboxes = torch.clamp(bboxes, 0)
    
    
    print(images_whwh[0])
    #bboxes[:, 0] /= w
    #bboxes[:, 2] /= w
    #bboxes[:, 1] /= h
    #bboxes[:, 3] /= h
    bboxes = bboxes.data.cpu().numpy()
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    
    fig1 = plt.figure()
    plt.axis('off')
    ax1 = fig1.add_subplot()
    
    if ori_images is not None:
        #print(ori_images, ori_images.shape)
        im = ori_images[0].data.cpu().numpy() + rgb_mean
        im = im.astype(np.uint8)
        im = im.transpose((1, 2, 0))
        plt.imshow(im)
    
    for i in range(len(bboxes)):
        if i % 2 == 0: color = 'r'
        else: color = 'orange'
        b = bboxes[i]
        rect = plt.Rectangle((b[0], b[1]),b[2],b[3], \
            alpha=1.0, linewidth=1.0, edgecolor=color, facecolor='none')
        #    alpha=0.3, linewidth=0.7, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        #plt.gca().add_patch(rect)
    plt.show()
    fig1.savefig(fname+'rect{}.png'.format(idx))
    plt.close(fig1)
    
def query_similar(query):
    N = query.shape[0]
    
    fig1 = plt.figure()
    
    ans = torch.matmul(query, query.t())
    ans = F.softmax(ans, -1)
    ans = ans.data.cpu().numpy()
    
    plt.matshow(ans)
    plt.show()
    plt.savefig('query_similar.png', bbox_inches='tight')
    plt.close(fig1)
    
    x,y = np.where(ans > 0.1)
    print(ans[x,y])
    print(ans)

def batch_box_iou(boxes1, boxes2):
    """
        boxes1: [B, N, 4]
        boxes2: [B, M, 4]
        
        return: [B, N, M]
    """
    area1 = (boxes1[:, :, 3] - boxes1[:, :, 1] + 1) * (boxes1[:, :, 2] - boxes1[:, :, 0] + 1)
    area2 = (boxes2[:, :, 3] - boxes2[:, :, 1] + 1) * (boxes2[:, :, 2] - boxes2[:, :, 0] + 1)
    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2].type(boxes1.dtype))  # [B, N,M,2]
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:].type(boxes1.dtype))  # [B, N,M,2]
    wh = (rb - lt).clamp(min=0)  # [B, N,M,2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [B, N,M]
    union = area1[:, :, None] + area2[:, None, :] - inter
    iou = inter / union
    return iou, union

def triplet_similar(ent_cls, bboxes):
    bboxes = bboxes.view(bboxes.shape[0], -1, 8)
    N, nr_boxes = bboxes.shape[:2]
    ent_cls = ent_cls.view(N, nr_boxes, -1)
    num_ent_class = ent_cls.shape[-1] // 2
    iou_feat_s = 1-batch_box_iou(bboxes[:, :, :4], bboxes[:, :, :4])[0].view(N, nr_boxes, nr_boxes)
    iou_feat_o = 1-batch_box_iou(bboxes[:, :, 4:], bboxes[:, :, 4:])[0].view(N, nr_boxes, nr_boxes)
    ent_cls = F.sigmoid(ent_cls)
    X,Y = np.meshgrid(np.arange(nr_boxes), np.arange(nr_boxes), indexing='ij')
    X,Y = X.reshape(-1), Y.reshape(-1)
    kl_div_s, kl_div_o = [], []
    for i in range(N):
        logp_x = ent_cls[i,X,:num_ent_class].log()
        p_x = ent_cls[i,Y,:num_ent_class]
        #kl_s = F.kl_div(logp_x, p_x, reduction='none').sum(-1).view(nr_boxes, nr_boxes)
        kl_s = F.l1_loss(ent_cls[i,X,:num_ent_class], ent_cls[i,Y,:num_ent_class], reduction='none').sum(-1).view(nr_boxes, nr_boxes)
        kl_div_s.append(kl_s)
        logp_x = ent_cls[i,X,num_ent_class:].log()
        p_x = ent_cls[i,Y,num_ent_class:]
        #kl_o = F.kl_div(logp_x, p_x, reduction='none').sum(-1).view(nr_boxes, nr_boxes)
        kl_o = F.l1_loss(ent_cls[i,X,num_ent_class:], ent_cls[i,Y,num_ent_class:], reduction='none').sum(-1).view(nr_boxes, nr_boxes)
        kl_div_o.append(kl_o)
    kl_div_s = torch.stack(kl_div_s)
    kl_div_o = torch.stack(kl_div_o)
    C_s = kl_div_s * 2 + iou_feat_s * 2
    C_o = kl_div_o * 2 + iou_feat_o * 2
    C = torch.maximum(C_s, C_o)
    
    C = C.view(nr_boxes, nr_boxes)
    
    
    ans = F.softmax(C, -1)
    ans = ans.data.cpu().numpy()
    fig1 = plt.figure()
    plt.matshow(ans)
    plt.show()
    plt.savefig('triplet_similar.png', bbox_inches='tight')
    plt.close(fig1)
    print(ent_cls)
    print(bboxes)
    print(ans)
    return C