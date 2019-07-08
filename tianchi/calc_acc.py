#coding=utf-8
import sys 
from dbengine import DBEngine
import numpy as np 
query_gt = [{"agg": [0], "cond_conn_op": 1, "sel": [0], "conds": [[2, 0, "50"], [3, 0, "5"]]}]
pred_queries =  [{"agg": [0], "cond_conn_op": 1, "sel": [0], "conds": [[2, 0, "50"], [3, 0, "5"]]}]
sql_data = [[]]   # 这里先给几个测试吧

table_ids = ['69cc8c0c334311e98692542696d6e445']
# db_path = '/home/yinshuai/code/code/nlp/nlp_competition/tianchi/nl2sql_baseline/data/val/val.db'
# db_path = '/home/yinshuai/code/code/nlp/nlp_competition/tianchi/nl2sql_baseline/data/val/val.db'


def execute_accuracy(query_gt, pred_queries, table_ids, db_path, sql_data):
    """
        Execution Accuracy 执行精确性，只要sql的执行结果一致就行

    """
    engine = DBEngine(db_path)
    ex_acc_num = 0
    for sql_gt, sql_pred, tid in zip(query_gt, pred_queries, table_ids):
        ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])

        try:
            ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], sql_pred['cond_conn_op'])
        except Exception as  e:
            print(e)
            ret_pred = None
        ex_acc_num += (ret_gt == ret_pred)
    print('\nexecute acc is {}'.format(ex_acc_num / len(sql_data)))
    return  ex_acc_num / len(sql_data)


# print(execute_accuracy(query_gt, pred_queries, table_ids, db_path, sql_data))

import os
import json 
data_path = './data'
fh_where_val = open(os.path.join(data_path, 'where_val_error.log'), 'w')
fh_where_oper = open(os.path.join(data_path, 'where_oper_error.log'), 'w')
fh_where_col = open(os.path.join(data_path, 'where_col_error.log'), 'w')
fh_where_cnt = open(os.path.join(data_path, 'where_cnt_error.log'), 'w')



# # pred_queries = [{"agg": [0], "cond_conn_op": 1, "sel": [0], "conds": [[2, 0, "50"], [3, 0, "5"]]}]
# pred_queries = [{"agg": [0], "cond_conn_op": 1, "sel": [1], "conds": [[2, 0, "50"], [3, 0, "5"]]}]

# pred_queries = [{"agg": [0], "cond_conn_op": 1, "sel": [1], "conds": [(2, 0, "50"), (3, 0, "5")]}] # for test 
# gt_queries =  [{"agg": [0], "cond_conn_op": 1, "sel": [0], "conds": [[2, 0, "50"], [3, 0, "5"]]}]
def check_part_acc( pred_queries):
    """
        判断各个组件的精确度
        param: 
                pred_queries: array of query
                gt_queries: array of query
                data: data在这里主要是用作复盘
        ouput: xxx 

          
    """
    tot_err = sel_num_err = agg_err = sel_err = 0.0
    cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
    for query in pred_queries:
        pred_qry = query['sql_pred']
        gt_qry = query['sql']
        # for easy test
        good = True
        sel_pred, agg_pred, where_rela_pred = pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op']
        sel_gt, agg_gt, where_rela_gt = gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op']

        if where_rela_gt != where_rela_pred:
            good = False
            cond_rela_err += 1

        if len(sel_pred) != len(sel_gt):
            good = False
            sel_num_err += 1

        pred_sel_dict = {k:v for k,v in zip(list(sel_pred), list(agg_pred))}
        gt_sel_dict = {k:v for k,v in zip(sel_gt, agg_gt)}
        if set(sel_pred) != set(sel_gt):
            good = False
            sel_err += 1
        agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
        agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
        if agg_pred != agg_gt:
            good = False
            agg_err += 1

        cond_pred = pred_qry['conds']
        cond_gt = gt_qry['conds']
        if len(cond_pred) != len(cond_gt):
            good = False
            cond_num_err += 1
            fh_where_cnt.write(
                        json.dumps(query, ensure_ascii=False).encode('utf-8') + '\n')
        else:
            cond_op_pred, cond_op_gt = {}, {}
            cond_val_pred, cond_val_gt = {}, {}
            for p, g in zip(cond_pred, cond_gt):
                cond_op_pred[p[0]] = p[1]
                cond_val_pred[p[0]] = p[2]
                cond_op_gt[g[0]] = g[1]
                cond_val_gt[g[0]] = g[2]

            if set(cond_op_pred.keys()) != set(cond_op_gt.keys()):
                cond_col_err += 1
                good = False
                fh_where_col.write(
                        json.dumps(query, ensure_ascii=False).encode('utf-8') + '\n')

            where_op_pred = [cond_op_pred[x] for x in sorted(cond_op_pred.keys())]
            where_op_gt = [cond_op_gt[x] for x in sorted(cond_op_gt.keys())]
            if where_op_pred != where_op_gt:
                cond_op_err += 1
                good = False
                fh_where_oper.write(
                        json.dumps(query, ensure_ascii=False).encode('utf-8') + '\n')

            where_val_pred = [cond_val_pred[x] for x in sorted(cond_val_pred.keys())]
            where_val_gt = [cond_val_gt[x] for x in sorted(cond_val_gt.keys())]
            if where_val_pred != where_val_gt:
                cond_val_err += 1
                good = False
                fh_where_val.write(
                        json.dumps(query, ensure_ascii=False).encode('utf-8') + '\n')
                

        if not good:
            tot_err += 1
    q_len = len(pred_queries) # 获取所有的个数
    print('Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f, total_err: %.3f'
                % (sel_num_err / q_len, sel_err / q_len, agg_err / q_len,
                   cond_num_err / q_len, cond_col_err / q_len, cond_op_err / q_len,
                   cond_val_err / q_len, cond_rela_err / q_len, tot_err / q_len))
    return np.array((sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err, cond_rela_err)), tot_err  # 这里返回的都是总数哦，需要特殊处理成占比

res =  check_part_acc(pred_queries, gt_queries)
print(res)


