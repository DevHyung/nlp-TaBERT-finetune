import subprocess
import os
from openpyxl import load_workbook


class TREC_evaluator(object):

    def __init__(self,run_id,base_path = "/workspace/MAC_TABERT/TaBERT",trec_cmd = "./trec_eval"):
        self.run_id = run_id
        self.base_path = base_path
        self.rank_path = os.path.join(self.base_path ,self.run_id + ".result")
        self.qrel_path = os.path.join(self.base_path ,self.run_id + ".qrels")
        self.trec_cmd = trec_cmd

    def get_ndcgs(self, metric='ndcg_cut', qrel_path=None, rank_path=None, all_queries=False):
        if qrel_path is None:
            qrel_path = self.qrel_path
        if rank_path is None:
            rank_path = self.rank_path

        metrics = ['ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_15', 'ndcg_cut_20',
                   # 'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20',
                   'map', 'recip_rank']  # 'relstring'
        if all_queries:
            results = subprocess.run([self.trec_cmd, '-c', '-m', metric, '-q', qrel_path, rank_path],
                                     stdout=subprocess.PIPE).stdout.decode('utf-8')
            q_metric_dict = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')

                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics:
                    continue
                if metric != 'relstring':
                    score = float(seps[2].strip())
                else:
                    score = seps[2].strip()
                if qid not in q_metric_dict:
                    q_metric_dict[qid] = dict()
                q_metric_dict[qid][metric] = score
            return q_metric_dict

        else:
            results = subprocess.run([self.trec_cmd, '-c', '-m', metric, qrel_path, rank_path],
                                     stdout=subprocess.PIPE).stdout.decode('utf-8')

            ndcg_scores = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')
                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics or qid != 'all':
                    continue
                ndcg_scores[seps[0].strip()] = float(seps[2])
            return ndcg_scores

def load_excel(_fileName) -> dict:
    # Load Excel
    isFirst = True
    tEx = load_workbook(filename=_fileName)
    sheet1 = tEx.active
    dataDict = {}
    for i in sheet1.rows:
        if isFirst:
            isFirst = False
            continue

        qid = str(i[0].value)
        query = i[1].value
        tid = i[2].value
        rel = i[3].value
        cos_concat = i[4].value
        if cos_concat == '-': # 이부분이 조금 걸리는데 - 부분을 0으로하는거, 뺴는거 다시 재보자
            cos_concat = 0
            #continue
        else:
            cos_concat = float(cos_concat)

        t = (qid, query, tid, rel, cos_concat, i[5].value, i[6].value)
        try:
            dataDict[qid].append(t)
        except:
            dataDict[qid] = [t]

    return dataDict

if __name__ == "__main__":
    """
    epochNum = 2
    dataDict = load_excel('epoch{}_batch64_m1_y1.xlsx'.format(epochNum))
    #dataDict = load_excel('epoch{}_batch64_m1_y1_lossformal.xlsx'.format(epochNum))
    # 소팅해서
    # 필요  무시 필요 무시 필요 무시
    # query-id "0" document-id rank score STANDARD
    f = open('./test.result', 'w')
    lineFormat = "{}\t0\t{}\t{}\t{}\ttest\n"
    for k in dataDict.keys():
        resultList = dataDict[k]
    
        # True List
        relSortedList = sorted(resultList, key=lambda resultList: resultList[3], reverse=True)
        for line in relSortedList:
            f.write(lineFormat.format(line[0], line[2], line[3], line[4]))
    f.close()
    print("Succes test.result file")
    """
    trec_eval = TREC_evaluator(run_id="test")
    #print(epochNum, trec_eval.get_ndcgs())
    print(trec_eval.get_ndcgs())

# 원본 0.2265	0.2614	0.2839	0.3235

# 우리 Loss
# 0 {'ndcg_cut_5': 0.2599, 'ndcg_cut_10': 0.2745, 'ndcg_cut_15': 0.3059, 'ndcg_cut_20': 0.3423}
# 1 {'ndcg_cut_5': 0.2591, 'ndcg_cut_10': 0.2701, 'ndcg_cut_15': 0.3065, 'ndcg_cut_20': 0.3503}
#!2 {'ndcg_cut_5': 0.2699, 'ndcg_cut_10': 0.2954, 'ndcg_cut_15': 0.3412, 'ndcg_cut_20': 0.3734}
# 3 {'ndcg_cut_5': 0.2559, 'ndcg_cut_10': 0.2841, 'ndcg_cut_15': 0.3133, 'ndcg_cut_20': 0.3425}
# 4 {'ndcg_cut_5': 0.2538, 'ndcg_cut_10': 0.2769, 'ndcg_cut_15': 0.317, 'ndcg_cut_20': 0.3441}
# 5 {'ndcg_cut_5': 0.2266, 'ndcg_cut_10': 0.2752, 'ndcg_cut_15': 0.3158, 'ndcg_cut_20': 0.3475}
# 6 {'ndcg_cut_5': 0.244, 'ndcg_cut_10': 0.2819, 'ndcg_cut_15': 0.312, 'ndcg_cut_20': 0.3499}
# 7 {'ndcg_cut_5': 0.2415, 'ndcg_cut_10': 0.2712, 'ndcg_cut_15': 0.3059, 'ndcg_cut_20': 0.3357}
# 8 {'ndcg_cut_5': 0.2238, 'ndcg_cut_10': 0.2604, 'ndcg_cut_15': 0.2898, 'ndcg_cut_20': 0.3324}
# 9 {'ndcg_cut_5': 0.2472, 'ndcg_cut_10': 0.2719, 'ndcg_cut_15': 0.3095, 'ndcg_cut_20': 0.341}
# 15 {'ndcg_cut_5': 0.2419, 'ndcg_cut_10': 0.2765, 'ndcg_cut_15': 0.3001, 'ndcg_cut_20': 0.3324}
# 19 {'ndcg_cut_5': 0.2566, 'ndcg_cut_10': 0.2778, 'ndcg_cut_15': 0.3117, 'ndcg_cut_20': 0.3435}



# nn.Loss
# 0 {'ndcg_cut_5': 0.2265, 'ndcg_cut_10': 0.256, 'ndcg_cut_15': 0.2814, 'ndcg_cut_20': 0.316}
# 1 {'ndcg_cut_5': 0.2373, 'ndcg_cut_10': 0.2793, 'ndcg_cut_15': 0.315, 'ndcg_cut_20': 0.351}
# 2 {'ndcg_cut_5': 0.2436, 'ndcg_cut_10': 0.2663, 'ndcg_cut_15': 0.3017, 'ndcg_cut_20': 0.3319}
# 3 {'ndcg_cut_5': 0.2341, 'ndcg_cut_10': 0.2525, 'ndcg_cut_15': 0.2962, 'ndcg_cut_20': 0.3367}
#!4 {'ndcg_cut_5': 0.2668, 'ndcg_cut_10': 0.3022, 'ndcg_cut_15': 0.3277, 'ndcg_cut_20': 0.3577}
# 5 {'ndcg_cut_5': 0.2583, 'ndcg_cut_10': 0.2845, 'ndcg_cut_15': 0.314, 'ndcg_cut_20': 0.3422}
# 6 {'ndcg_cut_5': 0.2504, 'ndcg_cut_10': 0.2638, 'ndcg_cut_15': 0.2958, 'ndcg_cut_20': 0.335}
# 8 {'ndcg_cut_5': 0.2388, 'ndcg_cut_10': 0.2804, 'ndcg_cut_15': 0.2957, 'ndcg_cut_20': 0.3294}


# 2 {'ndcg_cut_5': 0.2442, 'ndcg_cut_10': 0.2686, 'ndcg_cut_15': 0.3089, 'ndcg_cut_20': 0.3407}
# 3 {'ndcg_cut_5': 0.2492, 'ndcg_cut_10': 0.2801, 'ndcg_cut_15': 0.3045, 'ndcg_cut_20': 0.3357}
# 4 {'ndcg_cut_5': 0.2421, 'ndcg_cut_10': 0.2673, 'ndcg_cut_15': 0.2973, 'ndcg_cut_20': 0.3407}
# 5 {'ndcg_cut_5': 0.2918, 'ndcg_cut_10': 0.3087, 'ndcg_cut_15': 0.3345, 'ndcg_cut_20': 0.3673 }
# 6 {'ndcg_cut_5': 0.248, 'ndcg_cut_10': 0.2793, 'ndcg_cut_15': 0.3182, 'ndcg_cut_20': 0.3535}
# 7 {'ndcg_cut_5': 0.2518, 'ndcg_cut_10': 0.2834, 'ndcg_cut_15': 0.3126, 'ndcg_cut_20': 0.3392}
# 8 {'ndcg_cut_5': 0.2579, 'ndcg_cut_10': 0.2807, 'ndcg_cut_15': 0.3214, 'ndcg_cut_20': 0.3512}
# 9 {'ndcg_cut_5': 0.2347, 'ndcg_cut_10': 0.263, 'ndcg_cut_15': 0.2947, 'ndcg_cut_20': 0.3389}
# 15 {'ndcg_cut_5': 0.2715, 'ndcg_cut_10': 0.2901, 'ndcg_cut_15': 0.3232, 'ndcg_cut_20': 0.3508}
# 19 {'ndcg_cut_5': 0.2573, 'ndcg_cut_10': 0.2883, 'ndcg_cut_15': 0.3304, 'ndcg_cut_20': 0.3558}

