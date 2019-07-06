import json
old_version_file = './evaluate_pred'
new_version_file = './evaluate_pred_new'

f_o = open(old_version_file, 'r')
f_n = open(new_version_file, 'r')

for i, (d_o, d_n) in enumerate(zip(f_o, f_n)):
    d_o = json.loads(d_o)
    d_n = json.loads(d_n)
    if d_o['sql_pred'] != d_n['sql_pred']:
        print('-' * 6)
        print(d_o)
        # print(d_o[''])
        print(d_o['sql'])
        print(d_o['sql_pred'])
        print(d_n['sql_pred'])
