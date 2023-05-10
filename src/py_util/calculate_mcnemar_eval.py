import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

eval_base_path = "../../out/ustancebr/eval"
pred_base_path = "../../out/ustancebr/pred"
data_folder = "simple_domain"
out_path = f"{eval_base_path}/.results/data/bertimbau.VS.bertabaporu/mcnemar2.csv"
true_data_base_path = "../../v2/simple_domain/final_{topic}_test.csv"

version_to_compare_dict = [
    {
        "label_topic": "bo",
        "src_topic1":"bo",
        "model_name1": "BertBiLSTMAttn",
        "version1":"14",
        "src_topic2":"bo",
        "model_name2": "BertBiLSTMAttn",
        "version2":"54"},
    {
        "label_topic": "lu",
        "src_topic1":"lu",
        "model_name1": "BertBiLSTMAttn",
        "version1":"24",
        "src_topic2":"lu",
        "model_name2": "BertBiLSTMAttn",
        "version2":"56"
    },
    {
        "label_topic": "co",
        "src_topic1":"co",
        "model_name1": "BertBiLSTMAttn",
        "version1":"23",
        "src_topic2":"co",
        "model_name2": "BertBiLSTMAttn",
        "version2":"53"
    },
    {
        "label_topic": "cl",
        "src_topic1":"cl",
        "model_name1": "BertBiLSTMAttn",
        "version1":"23",
        "src_topic2":"cl",
        "model_name2": "BertBiLSTMAttn",
        "version2":"56"
    },
    {
        "label_topic": "gl",
        "src_topic1":"gl",
        "model_name1": "BertBiLSTMAttn",
        "version1":"21",
        "src_topic2":"gl",
        "model_name2": "BertBiLSTMAttn",
        "version2":"40"
    },
    {
        "label_topic": "ig",
        "src_topic1":"ig",
        "model_name1": "BertBiLSTMAttn",
        "version1":"22",
        "src_topic2":"ig",
        "model_name2": "BertBiLSTMAttn",
        "version2":"54"
    },
]

with open(out_path, "w") as f_:
    for version_dict in version_to_compare_dict:
        dest_topic = version_dict["label_topic"]
        src_topic1 = version_dict["src_topic1"]
        model_name1 = version_dict["model_name1"]
        version1 = version_dict["version1"]
        src_topic2 = version_dict["src_topic2"]
        version2 = version_dict["version2"]
        model_name2 = version_dict["model_name2"]
        
        v1_path = f"{pred_base_path}/{data_folder}/{model_name1}_{src_topic1}_{dest_topic}_BEST_v{version1}-test.csv"
        v2_path = f"{pred_base_path}/{data_folder}/{model_name2}_{src_topic2}_{dest_topic}_BEST_v{version2}-test.csv"

        v1_pred = pd.read_csv(v1_path).set_index("Text")
        v2_pred = pd.read_csv(v2_path).set_index("Text")
        
        v1_pred["match"] = v1_pred.eval("Polarity == Polarity_pred")
        v2_pred["match"] = v2_pred.eval("Polarity == Polarity_pred")

        # cria tabela de contingencia
        df = pd.concat([v1_pred["match"],v2_pred["match"]], axis=1) # lado
        df.columns = ['pre','pos']
        df['right_right'] = df.apply(lambda x: int(x.pre==1 and x.pos==1), axis=1)
        df['right_wrong'] = df.apply(lambda x: int(x.pre==1 and x.pos==0), axis=1)
        df['wrong_right'] = df.apply(lambda x: int(x.pre==0 and x.pos==1), axis=1)
        df['wrong_wrong'] = df.apply(lambda x: int(x.pre==0 and x.pos==0), axis=1)
        
        cell_11 = df.right_right.sum()
        cell_12 = df.right_wrong.sum()
        cell_21 = df.wrong_right.sum()
        cell_22 = df.wrong_wrong.sum()
        table = [
            [cell_11, cell_12],
            [cell_21, cell_22],
        ]

        if cell_11<25 or cell_12<25 or cell_21<25 or cell_22<25:
            result = mcnemar(table, exact=True)
        else:
            result = mcnemar(table, exact=False, correction=True)
            
        print(str(version_dict).replace("\n", " "), file=f_)
        print(result, file=f_)
    
    print(file=f_, flush=True)