import os
import json
import torch
import pickle as pkl
from model import ICLMModel
from sklearn import metrics
batch_size = 64
use_gpu = False


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

def init_matrix(matrix, kg, entity2id, relation2id, relation_tail):
    # print('Processing Matirx(shape={})'.format(matrix.shape))
    for triple in kg:
        h, r, t = triple.get_triple()
        if t not in relation_tail: continue
        entity_a = entity2id[h]
        entity_b = relation_tail[t]
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

def get_init_matrix(kg, entity2id, relation2id, triples):
    i_x = []
    i_y = []
    v = []
    cur_set = set()
    for x, r, y, x_hat, y_hat in triples:
        if r.startswith('INV'):
            cur_set.add('{}||{}||{}'.format(x, r, y))
            cur_set.add('{}||{}||{}'.format(y, r[3:], x))
        else:
            cur_set.add('{}||{}||{}'.format(x, r, y))
            cur_set.add('{}||{}||{}'.format(y, 'INV' + r, x))

    records = set()
    for triple in kg:
        h, r, t = triple.get_triple()
        flag = '{}||{}||{}'.format(h, r, t)
        if flag in cur_set: continue
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

def evaluate(relation, entity2id, id2relation, relation2id, train_kg, eval_kg, option, model_save_path):
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
    i, v = get_init_matrix(train_kg, entity2id, relation2id, [])
    matrix_all = torch.sparse.FloatTensor(i, v, torch.Size([len(entity2id) * (len(relation2id) + 1),
                                                            len(entity2id)]))
    if use_gpu: matrix_all = matrix_all.cuda()
    id2tail = {}
    tail2id = {}
    for h, t, label in eval_kg:
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
    for i in range(batch_num):
        cur_szie = batch_size
        if i == batch_num - 1:
            cur_szie = len(tail2id) - i * batch_size
        matrix = torch.zeros([len(entity2id), cur_szie, len(relation2id) + 1])
        entity2id_tail = {}
        for j in range(cur_szie):
            entity2id_tail[id2tail[i * batch_size + j]] = j
        init_matrix(matrix, train_kg, entity2id, relation2id, entity2id_tail)
        matrix = matrix.view(-1, len(relation2id) + 1).to_sparse()
        if use_gpu: matrix = matrix.cuda()
        all_states = []
        for step in range(option.step):
            state = model(matrix, matrix_all, all_states, step, entity2id_tail, flag=True)
            # print(state.sum())
            all_states.append(state)
        total_states.append(all_states[-1].cpu().detach())
    total_states = torch.cat(total_states, dim=-1)
    results = []
    for h, t, label in eval_kg:
        if h not in entity2id or t not in entity2id: continue
        truth_score = total_states[entity2id[h]][tail2id[t]]
        results.append((h, relation, t, label, truth_score.item()))
        if label != '0': print(h, relation, t, label, truth_score.item())
    return results

def search_thr(results):
    thrs = range(-500, 500)
    max_acc = 0
    best_thr = 0
    for thr in thrs:
        thr = thr / 100
        y_pred = []
        y_true = []
        for h, relation, t, label, score in results:
            pred = 0
            if score > thr: pred = 1
            y_pred.append(pred)
            y_true.append(int(label))
        acc = metrics.accuracy_score(y_true, y_pred)
        if acc > max_acc:
            max_acc = acc
            best_thr = thr
    return max_acc, best_thr

def apply_thr(results, thr):
    y_pred = []
    y_true = []
    for h, relation, t, label, score in results:
        pred = 0
        if score > thr: pred = 1
        y_pred.append(pred)
        y_true.append(int(label))
    return y_true, y_pred

if __name__ == '__main__':
    valid_data = {}
    test_data = {}
    dataset = 'FamilyIC'
    triples = set()
    entity2id = {}
    id2entity = {}

    with open('data/{}/entities.dict'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            idx, entity = line.strip().split('\t')
            entity2id[entity] = int(idx)
            id2entity[int(idx)] = entity

    with open('data/{}/train.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            triples.add('{}\t{}\t{}'.format(h, r, t))

    triples_valid = []
    with open('data/{}/valid.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            triple = '{}\t{}\t{}'.format(h, r, t)
            triples.add(triple)
            triples_valid.append(triple)
    triples_test = []
    with open('data/{}/test.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            triple = '{}\t{}\t{}'.format(h, r, t)
            triples.add(triple)
            triples_test.append(triple)
    triple_list_valid = set()
    for triple in triples_valid:
        h, r, t = triple.split('\t')
        triple = '{}\t{}\t{}'.format(h, r, t)
        triple_list_valid.add(triple + '\t1')
        for entity in entity2id:
            triple_corrupt_head = '{}\t{}\t{}'.format(entity, r, t)
            triple_corrupt_tail = '{}\t{}\t{}'.format(h, r, entity)
            if triple_corrupt_head not in triples: triple_list_valid.add(triple_corrupt_head + '\t0')
            if triple_corrupt_tail not in triples: triple_list_valid.add(triple_corrupt_tail + '\t0')

    triple_list_test = set()
    for triple in triples_test:
        h, r, t = triple.split('\t')
        triple = '{}\t{}\t{}'.format(h, r, t)
        triple_list_test.add(triple + '\t1')
        for entity in entity2id:
            triple_corrupt_head = '{}\t{}\t{}'.format(entity, r, t)
            triple_corrupt_tail = '{}\t{}\t{}'.format(h, r, entity)
            if triple_corrupt_head not in triples: triple_list_test.add(triple_corrupt_head + '\t0')
            if triple_corrupt_tail not in triples: triple_list_test.add(triple_corrupt_tail + '\t0')
    for triple in triple_list_valid:
        h, r, t, label = triple.split('\t')
        if r in valid_data:
            valid_data[r].append((t, h, label))
        else:
            valid_data[r] = [(t, h, label)]

    for triple in triple_list_test:
        h, r, t, label = triple.split('\t')
        if r in test_data:
            test_data[r].append((t, h, label))
        else:
            test_data[r] = [(t, h, label)]

    overall_acc_valid = 0
    thrs = {}
    for relation in valid_data:
        exp_dir = 'exps_{}/{}-{}-ori'.format(dataset, dataset, relation)
        option = Option(exp_dir)
        train_kg, entity2id, relation2id = load_kg_form_pkl('{}/'.format(exp_dir), relation.replace('/', '|'))
        # id2entity = reverse(entity2id)
        id2relation = reverse(relation2id)
        results_valid = evaluate(relation, entity2id, id2relation, relation2id, train_kg, valid_data[relation], option,
                 '{}/model_{}.pt'.format(exp_dir, relation.replace('/', '|'), relation.replace('/', '|')))
        best_acc, thr = search_thr(results_valid)
        thrs[relation] = thr
        overall_acc_valid += best_acc / len(valid_data)
        print(relation, thr, best_acc)

    y_true_overall = []
    y_pred_overall = []
    for relation in test_data:
        exp_dir = 'exps_{}/{}-{}-ori'.format(dataset, dataset, relation)
        option = Option(exp_dir)
        train_kg, entity2id, relation2id = load_kg_form_pkl('{}/'.format(exp_dir), relation.replace('/', '|'))
        # id2entity = reverse(entity2id)
        id2relation = reverse(relation2id)
        results_test = evaluate(relation, entity2id, id2relation, relation2id, train_kg, test_data[relation], option,
                                 '{}/model_{}.pt'.format(exp_dir, relation.replace('/', '|'),
                                                         relation.replace('/', '|')))
        thr = 0
        if relation in thrs: thr = thrs[relation]
        y_true, y_pred = apply_thr(results_test, thr)
        y_true_overall.extend(y_true)
        y_pred_overall.extend(y_pred)
    overall_acc = metrics.accuracy_score(y_true_overall, y_pred_overall)
    precision = metrics.precision_score(y_true_overall, y_pred_overall)
    recall = metrics.recall_score(y_true_overall, y_pred_overall)
    f1 = metrics.f1_score(y_true_overall, y_pred_overall)
    print(overall_acc, precision, recall, f1)
    print(metrics.classification_report(y_true_overall, y_pred_overall))
