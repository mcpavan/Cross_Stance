import argparse
import copy
import math
import time
import torch
import pandas as pd

import datasets, models, input_models, model_utils

SEED = 0
use_cuda = torch.cuda.is_available()

def load_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    return config

def load_data(config, args, data_key="trn"):
    if 'bert' in config or 'bert' in config['name']:
        data = datasets.BertStanceDataset(
            data_file = args[f'{data_key}_data'],
            pd_read_kwargs = {},
            text_col = config["text_col"],
            topic_col = config["topic_col"],
            label_col = config["label_col"],
            max_seq_len_text = config["max_seq_len_text"],
            max_seq_len_topic = config["max_seq_len_topic"],
            pad_value = config.get("pad_value"),
            add_special_tokens = config.get("add_special_tokens"),
            bert_pretrained_model = config.get("bert_pretrained_model"),
            is_joint = config.get("is_joint"),
            sample_weights = config.get("sample_weights"),
        )
    else:
        #TODO: Create dataset for non-BERT models
        data = None

    return data

def eval_helper(model_handler, data_name, data=None):
    '''
    Helper function for evaluating the model during training.
    Can evaluate on all the data or just a subset of corpora.
    :param model_handler: the holder for the model
    :return: the scores from running on all the data
    '''
    # eval on full corpus
    return model_handler.eval_and_print(data=data, data_name=data_name) #(score, avg_loss)

def train(model_handler, num_epochs, early_stopping_patience=0, verbose=True, vld_data=None, tst_data=None):
    '''
    Trains the given model using the given data for the specified
    number of epochs. Prints training loss and evaluation. Saves at
    most 1 checkpoint plus a final one.
    :param model_handler: a holder with a model and data to be trained.
                            Assuming the model is a pytorch model.
    :param num_epochs: the number of epochs to train the model for.
    :param verbose: whether or not to print train results while training.
                    Default (True): do print intermediate results.
    '''
    trn_scores_dict = {}
    vld_scores_dict = {}
    tst_scores_dict = {}
    
    best_vld_loss = float("inf")
    last_vld_loss = float("inf")
    greater_loss_epochs = 0

    for epoch in range(num_epochs):
        model_handler.train_step()
        # print training loss 
        print("Total training loss: {}".format(model_handler.loss))

        # print training (& vld) scores
        if verbose:
            # eval model on training data
            # trn_scores, trn_loss = eval_helper(
            #     model_handler,
            #     data_name='TRAIN'
            # )
            # trn_scores_dict[epoch] = copy.deepcopy(trn_scores)
            # print("Training loss: {}".format(trn_loss))

            # update best model checkpoint
            if vld_data is not None:
                vld_scores, vld_loss = eval_helper(
                    model_handler,
                    data_name='VALIDATION',
                    data=vld_data
                )
                vld_scores_dict[epoch] = copy.deepcopy(vld_scores)
                # print vld loss
                print("Avg Validation loss: {}".format(vld_loss))

                #check if is best vld loss to save best model
                if vld_loss < best_vld_loss:
                    model_handler.save_best()

                # check if the current vld loss is greater than the last loss and
                # break the training loop if its over the early stopping patience
                early_stopping_patience += vld_loss > last_vld_loss
                if greater_loss_epochs > early_stopping_patience:
                    break
                
                if model_handler.has_scheduler:
                    model_handler.scheduler.step(vld_loss)
            else:
                model_handler.save_best()
            
            if tst_data is not None:
                tst_scores, tst_loss = eval_helper(
                    model_handler,
                    data_name='TEST',
                    data=tst_data
                )
                tst_scores_dict[epoch] = copy.deepcopy(tst_scores)
                # print vld loss
                print("Avg Test loss: {}".format(tst_loss))

    print("TRAINED for {} epochs".format(epoch))

    # save final checkpoint
    model_handler.save(num="FINAL")

    # print final training (& dev) scores
    eval_helper(
        model_handler,
        data_name='TRAIN'
    )
    
    if vld_data is not None:
        eval_helper(
            model_handler,
            data_name='VALIDATION',
            data=vld_data
        )
    
    if tst_data is not None:
        eval_helper(
            model_handler,
            data_name='TEST',
            data=tst_data
        )

def save_predictions(model_handler, dev_data, dev_dataloader, out_name, config, is_test=False, is_valid=False):#, correct_preds=False):
    # trn_results = model_handler.predict()
    # trn_preds = trn_results[0]
    
    dev_results = model_handler.predict(data=dev_dataloader)#, correct_preds=correct_preds)
    # dev_preds = dev_results[0]

    dev_preds = []
    if int(config['n_output_classes']) == 2:
        for pred_val in dev_results[0]:
            int_pred = int(pred_val > 0.5)
            vec_pred = (int_pred,)

            dev_preds.append(dev_data.convert_vec_to_lbl(vec_pred))
    else:
        base_vec = [0 for i in range(int(config['n_output_classes']))]
        
        for pred_val in dev_results[0]:
            int_pred = pred_val.argmax()
            
            vec_pred = base_vec*0
            vec_pred[int_pred] = 1
            vec_pred = tuple(vec_pred)

            dev_preds.append(dev_data.convert_vec_to_lbl(vec_pred))

    if is_test:
        dev_name = 'test'
    elif is_valid:
        dev_name = 'dev'
    else:
        dev_name = 'train'

    # predict_helper(trn_preds, model_handler.dataloader).to_csv(out_name + '-train.csv', index=False)
    # print("saved to {}-train.csv".format(out_name))
    predict_helper(dev_preds, dev_data, config).to_csv(out_name + '-{}.csv'.format(dev_name), index=False)
    print("saved to {}-{}.csv".format(out_name, dev_name))

def predict_helper(pred_lst, pred_data, config):
    out_data = []
    cols = [
        config['text_col'],
        config['topic_col'],
        config['label_col'],
    ]
    for i in pred_data.df.index:
        row = pred_data.df.iloc[i]
        temp = [row[c] for c in cols]
        temp.append(pred_lst[i])
        out_data.append(temp)
    cols += [config['label_col']+'_pred']
    return pd.DataFrame(out_data, columns=cols)

def main(args):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    ####################
    # load config file #
    ####################
    config = load_config_file(args['config_file'])

    #############
    # LOAD DATA #
    #############
    # load training data
    batch_size = int(config['batch_size'])
    if args['trn_data'] is not None:
        train_data = load_data(config, args, data_key="trn")
        train_n_batches = math.ceil(len(train_data) / batch_size)
        print(f"# of Training instances: {len(train_data)}. Batch Size={batch_size}. # of batches: {train_n_batches}")

        trn_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=int(config['batch_size']),
            shuffle=True
        )
    else:
        if args["mode"] == "train":
            raise "Please use a dataset to train the model."
        trn_dataloader = None

    # load vld data if specified
    if args['vld_data'] is not None:
        vld_data = load_data(config, args, data_key="vld")
        vld_n_batches = math.ceil(len(vld_data) / batch_size)
        print(f"# of Validation instances: {len(vld_data)}. Batch Size={batch_size}. # of batches: {vld_n_batches}")

        vld_dataloader = torch.utils.data.DataLoader(
            vld_data,
            batch_size=int(config['batch_size']),
            shuffle=False
        )
    else:
        vld_dataloader = None

    # load tst data if specified
    if args['tst_data'] is not None:
        tst_data = load_data(config, args, data_key="tst")
        tst_n_batches = math.ceil(len(tst_data) / batch_size)
        print(f"# of Test instances: {len(tst_data)}. Batch Size={batch_size}. # of batches: {tst_n_batches}")

        tst_dataloader = torch.utils.data.DataLoader(
            tst_data,
            batch_size=int(config['batch_size']),
            shuffle=False
        )
    else:
        tst_dataloader = None

    lr = float(config.get('learning_rate', '0.001'))
    nl = int(config.get("n_output_classes", "3"))

    # RUN
    print("Using cuda?: {}".format(use_cuda))

    if "BiLSTMAttn" in config["name"]:
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )

        loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
        if nl < 3:
            loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)#ignore_index=nl)

        model = models.BiLSTMAttentionModel(
            lstm_text_input_dim=text_input_layer.dim,
            lstm_hidden_dim=int(config['lstm_hidden_dim']),
            lstm_num_layers=int(config.get("lstm_layers","1")),
            lstm_drop_prob=float(config.get("lstm_drop_prob", config.get('dropout', "0"))),
            attention_density=int(config['attention_density']),
            attention_heads=int(config['attention_heads']),
            attention_drop_prob=float(config.get("attention_drop_prob", config.get('dropout', "0"))),
            drop_prob=float(config.get('dropout', "0")),
            num_labels=nl,
            use_cuda=use_cuda
        )

        opt = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            patience=2,
        )

        kwargs = {
            'model': model,
            'text_input_model': text_input_layer,
            'dataloader': trn_dataloader,
            'name': config['name'] + args['name'],
            'loss_function': loss_fn,
            'optimizer': opt,
            'scheduler': scheduler,
            'is_joint_text_topic':config.get("is_joint"),
        }

        model_handler = model_utils.TorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )
    
    elif "BiLSTMJointAttn" in config["name"]:
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )

        topic_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )
        
        loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
        if nl < 3:
            loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)#ignore_index=nl)

        model = models.BiLSTMJointAttentionModel(
            lstm_text_input_dim=text_input_layer.dim,
            lstm_topic_input_dim=topic_input_layer.dim,
            lstm_hidden_dim=int(config['lstm_hidden_dim']),
            lstm_num_layers=int(config.get("lstm_layers","1")),
            lstm_drop_prob=float(config.get("lstm_drop_prob", config.get('dropout', "0"))),
            attention_density=int(config.get('attention_density', None)),
            attention_heads=int(config['attention_heads']),
            attention_drop_prob=float(config.get("attention_drop_prob", config.get('dropout', "0"))),
            drop_prob=float(config.get('dropout', "0")),
            num_labels=nl,
            use_cuda=use_cuda
        )

        opt = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            patience=2,
        )

        kwargs = {
            'model': model,
            'text_input_model': text_input_layer,
            'topic_input_model': topic_input_layer,
            'dataloader': trn_dataloader,
            'name': config['name'] + args['name'],
            'loss_function': loss_fn,
            'optimizer': opt,
            'scheduler': scheduler,
            'is_joint_text_topic':config.get("is_joint"),
        }

        model_handler = model_utils.TorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )
    
    elif 'BiCond' in config['name']:
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )

        topic_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )
        
        loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
        if nl < 3:
            loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)

        model = models.BiCondLSTMModel(
            hidden_dim=int(config['lstm_hidden_dim']),
            text_input_dim=text_input_layer.dim,
            topic_input_dim=topic_input_layer.dim,
            num_layers=int(config.get("lstm_layers","1")),
            drop_prob=float(config.get('dropout', "0")),
            num_labels=nl,
            use_cuda=use_cuda,
        )

        opt = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            patience=2,
        )

        kwargs = {
            'model': model,
            'text_input_model': text_input_layer,
            'topic_input_model': topic_input_layer,
            'dataloader': trn_dataloader,
            'name': config['name'] + args['name'],
            'loss_function': loss_fn,
            'optimizer': opt,
            'scheduler': scheduler,
            'is_joint_text_topic':config.get("is_joint"),
        }

        model_handler = model_utils.TorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )

    # elif "mixture" in config["name"].lower():
    #     text_input_layer = input_models.BertLayer(
    #         use_cuda=use_cuda,
    #         pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
    #         layers=config.get("bert_layers", "-1"),
    #         layers_agg_type=config.get("bert_layers_agg", "concat"),
    #     )

    #     topic_input_layer = input_models.BertLayer(
    #         use_cuda=use_cuda,
    #         pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
    #         layers=config.get("bert_layers", "-1"),
    #         layers_agg_type=config.get("bert_layers_agg", "concat"),
    #     )
        
    #     loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
    #     if nl < 3:
    #         loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
    #     else:
    #         loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)#ignore_index=nl)
        
    #     base_params_dict = {
    #         "text_input_dim": text_input_layer.dim,
    #         "topic_input_dim": topic_input_layer.dim,
    #     }

    #     n_experts = int(config.get("n_experts", 3))
    #     expert_params = [copy.deepcopy(base_params_dict) for _ in range(n_experts)]
    #     gating_params = copy.deepcopy(base_params_dict)
        
    #     for key, value in config.items():
    #         if key.startswith("gating_"):
    #             gating_params[key.replace("gating_", "")] = value
    #         elif key.startswith("expert_"):
    #             expert_number = int(key.split("_")[1])
    #             expert_params[expert_number-1][key.replace(f"expert_{expert_number}_", "")] = value

    #     model = models.MixtureOfExpertsModel(
    #         n_experts=config.get("n_experts", 3),
    #         experts_params=expert_params,
    #         gating_params=gating_params,
    #         num_labels=nl,
    #         use_cuda=use_cuda,
    #     )

    #     o = torch.optim.Adam(
    #         model.parameters(),
    #         lr=lr
    #     )

    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer=o,
    #         patience=2,
    #     )

    #     kwargs = {
    #         'model': model,
    #         'text_input_model': text_input_layer,
    #         'topic_input_model': topic_input_layer,
    #         'dataloader': trn_dataloader,
    #         'name': config['name'] + args['name'],
    #         'loss_function': loss_fn,
    #         'optimizer': o,
    #         'scheduler': scheduler,
    #         'is_joint_text_topic':config.get("is_joint"),
    #     }

    #     model_handler = model_utils.TorchModelHandler(
    #         checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
    #         use_cuda=use_cuda,
    #         **kwargs
    #     )

    elif 'CrossNet' in config['name']:
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )

        topic_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )
        
        loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
        if nl < 3:
            loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)
        
        model = models.CrossNet(
            hidden_dim=int(config['lstm_hidden_dim']),
            attn_dim=int(config["attention_dimension"]),
            text_input_dim=text_input_layer.dim,
            topic_input_dim=topic_input_layer.dim,
            num_layers=int(config.get("lstm_layers","1")),
            drop_prob=float(config.get('dropout', "0")),
            num_labels=nl,
            use_cuda=use_cuda,
        )

        opt = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            patience=2,
        )

        kwargs = {
            'model': model,
            'text_input_model': text_input_layer,
            'topic_input_model': topic_input_layer,
            'dataloader': trn_dataloader,
            'name': config['name'] + args['name'],
            'loss_function': loss_fn,
            'optimizer': opt,
            'scheduler': scheduler,
            'is_joint_text_topic':config.get("is_joint"),
        }

        model_handler = model_utils.TorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )

    if args["mode"] == 'train':
        # Train model
        start_time = time.time()
        train(
            model_handler=model_handler,
            num_epochs=int(config['epochs']),
            early_stopping_patience=args.get("early_stop", 0),
            verbose=True,
            vld_data=vld_dataloader,
            tst_data=tst_dataloader,
        )
        print(f"[{config['name']}] total runtime: {(time.time() - start_time)/60:.2f} minutes")

    elif args["mode"] == 'eval':
        # Evaluate saved model
        model_handler.load(filename=args["saved_model_file_name"])
        if vld_dataloader is not None:
            eval_helper(
                model_handler,
                data_name='VALIDATION',
                data=vld_dataloader
            )

        if tst_dataloader is not None:
            eval_helper(
                model_handler,
                data_name='TEST',
                data=tst_dataloader
            )

    elif args["mode"] == "predict":
        # laod saved model and save the predictions
        model_handler.load(filename=args["saved_model_file_name"])
        
        if trn_dataloader is not None:
            save_predictions(
                model_handler=model_handler,
                dev_data=train_data,
                dev_dataloader=trn_dataloader,
                out_name=args["out_path"],
                config=config,
                is_test=False,#('test' in args["trn_data"] or 'tst' in args["trn_data"]),
                is_valid=False,#('valid' in args["trn_data"] or 'vld' in args["trn_data"]),
            )
            
        if vld_dataloader is not None:
            save_predictions(
                model_handler=model_handler,
                dev_data=vld_data,
                dev_dataloader=vld_dataloader,
                out_name=args["out_path"],
                config=config,
                is_test=False,#('test' in args["vld_data"] or 'tst' in args["vld_data"]),
                is_valid=True,#('valid' in args["vld_data"] or 'vld' in args["vld_data"]),
            )
        
        if tst_dataloader is not None:
            save_predictions(
                model_handler=model_handler,
                dev_data=tst_data,
                dev_dataloader=tst_dataloader,
                out_name=args["out_path"],
                config=config,
                is_test=True,#('test' in args["tst_data"] or 'tst' in args["tst_data"]),
                is_valid=False,#('valid' in args["tst_data"] or 'vld' in args["tst_data"]),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest="mode", help='What to do', required=True)
    parser.add_argument('-c', '--config_file', dest="config_file", help='Name of the cofig data file', required=False)
    parser.add_argument('-t', '--trn_data', dest="trn_data", help='Name of the training data file', required=False)
    parser.add_argument('-v', '--vld_data', dest="vld_data", help='Name of the validation data file', default=None, required=False)
    parser.add_argument('-p', '--tst_data', dest="tst_data", help='Name of the test data file', default=None, required=False)
    parser.add_argument('-n', '--name', dest="name", help='something to add to the saved model name', required=False, default='')
    parser.add_argument('-e', '--early_stop', dest="early_stop", help='Whether to do early stopping or not', required=False, type=bool, default=False)
    # parser.add_argument('-w', '--num_warm', dest="num_warm", help='Number of warm-up epochs', required=False, type=int, default=0)
    # parser.add_argument('-k', '--score_key', dest="score_key", help='Score to use for optimization', required=False, default='f_macro')
    parser.add_argument('-s', '--save_ckp', dest="save_ckp", help='Whether to save checkpoints', required=False, default=0, type=int)
    parser.add_argument('-f', '--saved_model_file_name', dest="saved_model_file_name", required=False, default=None)
    parser.add_argument('-o', '--out_path', dest="out_path", help='Ouput file name', default='./pred')
    args = vars(parser.parse_args())

    main(args)

# running

## LOCAL
# train BiCondBertLstm
# python train_model.py -m train -c ../../config/BiCondBertLstm_example.txt -t ../../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -n bo -e 5 -s 1
# python train_model.py -m train -c ../../config/BiCondBertLstm_example.txt -t ../../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -v ../../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1


# train BertBiLSTMJointAttn
# python train_model.py -m train -c ../../config/Bert_BiLstmJointAttn_example.txt -t ../../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -p ../../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1
# python train_model.py -m train -c ../../config/Bert_BiLstmJointAttn_example.txt -t ../../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -v ../../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -p ../../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1

# train BertBiLSTMAttn (Simple Domain)
# python train_model.py -m train -c ../../config/Bert_BiLstmAttn_example.txt -t ../../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 5 -s 1

# predict BertBiLSTMAttn (Simple Domain)
# python train_model.py -m predict -c ../../config/Bert_BiLstmAttn_example.txt  -p ../../../data/ustancebr/v2/simple_domain/final_bo_test.csv -f ../../checkpoints/ustancebr/V0/ckp-BertBiLSTMAttn_ustancebr_bo-BEST.tar -o ../../out/ustancebr/pred/BertBiLSTMAttn_bo_BEST_v0
# -t ../../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../../data/ustancebr/v2/simple_domain/final_bo_valid.csv

# eval BertBiLSTMAttn (Simple Domain)
# python train_model.py -m eval -c ../../config/simple_domain/Bert_BiLSTMAttn_v1.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -f ../../checkpoints/ustancebr/simple_domain/V1/ckp-BertBiLSTMAttn_ustancebr_bo-BEST.tar -o ../../out/ustancebr/eval/BertBiLSTMAttn_bo_BEST_v1 -t ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv

# train MixtureOfExperts (Simple Domain)
# python train_model.py -m train -c ../../config/Bert_Mixture_example.txt -t ../../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 5 -s 1

# train CrossNet (Simple Domain)
# python train_model.py -m train -c ../../config/Bert_CrossNet_example.txt -t ../../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 2 -s 1

## VM
# train BiCondBertLstm
# nohup python train_model.py -m train -c ../../config/Bert_BiCondLstm_example_v0.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -n bo -e 5 -s 1 &
# nohup python train_model.py -m train -c ../../config/Bert_BiCondLstm_example_v1.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -n bo -e 5 -s 1 &

# train BertBiLSTMJointAttn
# nohup python train_model.py -m train -c ../../config/Bert_BiLstmJointAttn_example_v0.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -p ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1 &
# nohup python train_model.py -m train -c ../../config/Bert_BiLstmJointAttn_example_v1.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -p ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1 &

# train BertBiLSTMAttn (Simple Domain)
# nohup python train_model.py -m train -c ../../config/Bert_BiLstmAttn_example.txt -t ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 5 -s 1 &
# nohup python train_model.py -m train -c ../../config/Bert_BiLstmAttn_example.txt -t ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 5 -s 1 &
