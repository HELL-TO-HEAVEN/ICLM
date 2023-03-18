import os
import json
import sys
import torch
import pickle as pkl
from model import ICLMModel
from tqdm import tqdm

batch_size = int(sys.argv[2])
inverse = int(sys.argv[3])
if inverse == 1:
    inverse = True
else:
    inverse = False
print(inverse)
use_gpu = False
if use_gpu: os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]


class Option(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)

def load_kg_form_pkl(file_path, target_relation):
    with open(file_path + 'kg_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        kg = pkl.load(fd)
    with open(file_path + 'entity2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        entity2id = pkl.load(fd)
    with open(file_path + 'relation2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        relation2id = pkl.load(fd)
    return kg, entity2id, relation2id

def save_tail(tail2id, flag, file_path=None):
    if file_path is None: return
    with open(os.path.join(file_path, 'tail2id_{}.pkl'.format(flag)), mode='wb') as fw:
        pkl.dump(tail2id, fw)

def load_kg(kg_file):
    kg = []
    with open(kg_file, mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            if len(items) != 3: continue
            h, r, t = items
            kg.append((h, r, t))
    return kg

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

def build_graph(kg, target_relation):
    graph = {}
    graph_entity = {}
    for triple in kg:
        h, r, t = triple.get_triple()
        if h not in graph:
            graph[h] = {r: [t]}
            graph_entity[h] = {t: [r]}
        else:
            if r in graph[h]:
                graph[h][r].append(t)
            else:
                graph[h][r] = [t]

            if t in graph_entity[h]:
                graph_entity[h][t].append(r)
            else:
                graph_entity[h][t] = [r]
    return graph, graph_entity

def init_matrix(matrix, kg, entity2id, entity2id_tail, relation2id):
    print('Processing Matirx(shape={})'.format(matrix.shape))
    for triple in tqdm(kg):
        h, r, t = triple.get_triple()
        # if r == target_relation: continue
        if t not in entity2id_tail: continue
        entity_a = entity2id[h]
        entity_b = entity2id_tail[t]
        relation = relation2id[r]
        matrix[entity_a][entity_b][relation] = 1
        matrix[entity2id[t]][entity_b][len(relation2id)] = 1
    # matrix[:, :, len(relation2id)] = 1

def get_head(heads, kg):
    entity2id_head = {}
    id2entity_head = {}
    for h in heads:
        entity2id_head[h] = len(entity2id_head)
        id2entity_head[entity2id_head[h]] = h
    for h, r, t in kg:
        if h not in entity2id_head: continue
        if t in entity2id_head: continue
        entity2id_head[t] = len(entity2id_head)
        id2entity_head[entity2id_head[t]] = t
    return entity2id_head, id2entity_head

def get_init_matrix(kg, entity2id, relation2id):
    i_x = []
    i_y = []
    v = []
    records = set()
    for triple in tqdm(kg):
        h, r, t = triple.get_triple()
        entity_a = entity2id[h]
        entity_b = entity2id[t]
        relation = relation2id[r]
        record = '{}\t{}'.format(entity_a * (len(relation2id) + 1) + relation, entity_b)
        if record not in records:
            i_x.append(entity_a * (len(relation2id) + 1) + relation)
            i_y.append(entity_b)
            v.append(1)
            records.add(record)

        record = '{}\t{}'.format(entity_a * (len(relation2id) + 1) + len(relation2id), entity_a)
        if record not in records:
            i_x.append(entity_a * (len(relation2id) + 1) + len(relation2id))
            i_y.append(entity_a)
            v.append(1)
            records.add(record)
    return torch.LongTensor([i_x, i_y]), torch.FloatTensor(v)

def evaluate(id2entity, entity2id, id2relation, relation2id, train_kg, eval_kg, option, model_save_path):
    print('Entity Num:', len(entity2id))
    print('Relation Num:', len(relation2id))
    print('Train KG Size:', len(train_kg))
    print('Eval KG Size:', len(eval_kg))
    graph, graph_entity = build_graph(train_kg, option.target_relation)
    model = ICLMModel(len(relation2id), option.step, option.length,
                           len(entity2id), option.tau_1, option.tau_2, use_gpu)

    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    # for parameter in model.parameters():
    #     print(parameter)
    if use_gpu: model = model.cuda()
    model.eval()
    i, v = get_init_matrix(train_kg, entity2id, relation2id)
    matrix_all = torch.sparse.FloatTensor(i, v, torch.Size([len(entity2id) * (len(relation2id) + 1),
                                                            len(entity2id)]))
    if use_gpu: matrix_all = matrix_all.cuda()
    id2tail = {}
    tail2id = {}
    for h, r, t in tqdm(eval_kg):
        if r != option.target_relation: continue
        if inverse:
            if h not in tail2id:
                tail2id[h] = len(tail2id)
                id2tail[tail2id[h]] = h
        else:
            if t not in tail2id:
                tail2id[t] = len(tail2id)
                id2tail[tail2id[t]] = t
    print(option.target_relation, len(tail2id))
    if len(tail2id) == 0: exit(0)
    if len(tail2id) % batch_size == 0:
        batch_num = int(len(tail2id) / batch_size)
    else:
        batch_num = int(len(tail2id) / batch_size) + 1

    total_states = []
    for i in tqdm(range(batch_num)):
        cur_szie = batch_size
        if i == batch_num - 1:
            cur_szie = len(tail2id) - i * batch_size
        matrix = torch.zeros([len(entity2id), cur_szie, len(relation2id) + 1])
        entity2id_tail = {}
        for j in range(cur_szie):
            entity2id_tail[id2tail[i * batch_size + j]] = j
        init_matrix(matrix, train_kg, entity2id, entity2id_tail, relation2id)
        matrix = matrix.view(-1, len(relation2id) + 1).to_sparse()
        if use_gpu: matrix = matrix.cuda()
        all_states = []
        for step in range(option.step):
            state = model(matrix, matrix_all, all_states, step, entity2id_tail, flag=inverse)
            print(state.sum())
            all_states.append(state)
        total_states.append(all_states[-1].cpu().detach())
    total_states = torch.cat(total_states, dim=-1)
    flag = 'ori'
    if inverse: flag = 'inv'
    torch.save(total_states, '{}/state-{}.pt'.format(option.exp_dir, flag))
    save_tail(tail2id, flag, file_path=option.exp_dir)

if __name__ == '__main__':
    option = Option(sys.argv[1])
    train_kg, entity2id, relation2id = load_kg_form_pkl('{}/'.format(option.exp_dir), option.target_relation.replace('/', '|'))
    id2entity = reverse(entity2id)
    id2relation = reverse(relation2id)
    eval_kg = load_kg('{}/test.txt'.format(option.data_dir))
    evaluate(id2entity, entity2id, id2relation, relation2id, train_kg, eval_kg, option,
             '{}/model_{}.pt'.format(option.exp_dir, option.target_relation.replace('/', '|'), option.target_relation.replace('/', '|')))