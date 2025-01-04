import argparse
import copy
import itertools
import math
import time
import torch
import pandas as pd
import json
import os
import numpy as np
import re

import datasets, models, input_models, model_utils, loss_fn as lf, JointCL_loss, llms

SEED = 0
use_cuda = torch.cuda.is_available()

def load_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = ":".join(l.strip().split(":")[1:])
    return config

def get_config_dict_prefix(config_dict, prefix):
    params = {}
    for k,v in config_dict.items():
        if k.startswith(prefix):
            k = k.replace(prefix, "")
            if v.lower().strip() in ["true", "false"]:
                params[k] = v.lower().strip() == "true"
            else:
                is_number = re.match(r"\A\d+(.\d*)\Z|\A\d+e(-)*\d+\Z", v)
                if (is_number is not None) and (("." in v) or ("e" in v)):
                    params[k] = float(v)
                elif is_number is not None:
                    params[k] = int(v)
                else:
                    params[k] = v
    return params

def load_data(config, args, data_key="trn", trn_data=None):
    tokenizer_params = get_config_dict_prefix(
        config_dict=config,
        prefix="hf_tokenizer_"
    )
    
    if 'bert' in config or 'bert' in config['name'] or config.get("model_type", "").lower() == "bert":
        dataset_args = {
            "data_file": args[f'{data_key}_data'],
            "pd_read_kwargs": {},#"engine": "python"},
            "text_col": args.get(f"{data_key}_text_col") or config["text_col"],
            "topic_col": args.get(f"{data_key}_topic_col") or config["topic_col"],
            "label_col": args.get(f"{data_key}_label_col") or config["label_col"],
            "max_seq_len_text": config["max_seq_len_text"],
            "max_seq_len_topic": config["max_seq_len_topic"],
            "data_sample": float(args.get("train_data_sample", 1.0)) if data_key == "trn" else 1,
            "random_state": int(args.get("random_state", 123)),
            "tokenizer_params": tokenizer_params,
            "alpha_load_classes": int(config.get("alpha_load_classes", "0")),
            "ensemble_usescores": int(config.get("ensemble_usescores", "0")),
        }
        if "pad_value" in config:
            dataset_args["pad_value"] = config.get("pad_value")
        if "add_special_tokens" in config:
            dataset_args["add_special_tokens"] = config.get("add_special_tokens")
        if "bert_pretrained_model" in config:
            dataset_args["bert_pretrained_model"] = config.get("bert_pretrained_model")
        if "is_joint" in config:
            dataset_args["is_joint"] = config.get("is_joint")
        if "sample_weights" in config:
            dataset_args["sample_weights"] = config.get("sample_weights")
        if f"skip_rows_{data_key}" in args:
            dataset_args["skip_rows"] = args.get(f"skip_rows_{data_key}")
        if f"ensemble_clf1_pred_{data_key}" in args:
            dataset_args["ensemble_clf1_pred_file"] = args.get(f"ensemble_clf1_pred_{data_key}")
        if f"ensemble_clf2_pred_{data_key}" in args:
            dataset_args["ensemble_clf1_pred_file"] = args.get(f"ensemble_clf2_pred_{data_key}")

        data = datasets.BertStanceDataset(**dataset_args)

    elif config.get("model_type", "").lower() in ["llama_cpp", "hf_llm", "hf_api"]:
        dataset_args = {
            "data_file": args[f'{data_key}_data'],
            "prompt_file": config["prompt_template_file"],
            "pd_read_kwargs": {},#"engine": "python"},
            "text_col": args.get(f"{data_key}_text_col") or config["text_col"],
            "topic_col": args.get(f"{data_key}_topic_col") or config["topic_col"],
            "label_col": args.get(f"{data_key}_label_col") or config["label_col"],
            "data_sample": float(args.get("train_data_sample", 1.0)) if data_key == "trn" else 1,
            "random_state": int(args.get("random_state", 123)),
            "tokenizer_params": tokenizer_params,
            "alpha_load_classes": int(config.get("alpha_load_classes", "0")),
            "use_chat_template": int(config.get("use_chat_template", "0")),
        }

        needs_few_shot_examples = int(config.get("needs_few_shot_examples", "0"))
        if "max_seq_len_text" in config:
            dataset_args["max_seq_len_text"] = config.get("max_seq_len_text")
        if "n_texts_in_prompt" in config:
            dataset_args["n_texts_in_prompt"] = config.get("n_texts_in_prompt")
        if "max_seq_len_topic" in config:
            dataset_args["max_seq_len_topic"] = config.get("max_seq_len_topic")
        if "n_topics_in_prompt" in config:
            dataset_args["n_topics_in_prompt"] = config.get("n_topics_in_prompt")
        if "max_seq_len_examples" in config:
            dataset_args["max_seq_len_examples"] = config.get("max_seq_len_examples")
        if "n_examples_in_prompt" in config:
            dataset_args["n_examples_in_prompt"] = config.get("n_examples_in_prompt")
        if "pretrained_model_name" in config:
            dataset_args["pretrained_model_name"] = config.get("pretrained_model_name")
        if "sample_weights" in config:
            dataset_args["sample_weights"] = config.get("sample_weights")
        if "model_type" in config:
            dataset_args["model_type"] = config.get("model_type")
        if f"skip_rows_{data_key}" in args:
            dataset_args["skip_rows"] = args.get(f"skip_rows_{data_key}")
        if needs_few_shot_examples:
            if "sentence_model_name" in config:
                dataset_args["sentence_model_name"] = config.get("sentence_model_name")
            if "n_similar_sentences" in config:
                dataset_args["n_similar_sentences"] = config.get("n_similar_sentences")        
            dataset_args["train_dataset"] = trn_data

        data = datasets.LLMStanceDataset(**dataset_args)
    else:
        #TODO: Create dataset for other types of models
        data = None

    return data

def eval_helper(model_handler, data_name, data=None, y_pred=None):
    '''
    Helper function for evaluating the model during training.
    Can evaluate on all the data or just a subset of corpora.
    :param model_handler: the holder for the model
    :return: the scores from running on all the data
    '''
    # eval on full corpus
    return model_handler.eval_and_print(data=data, data_name=data_name, y_pred=y_pred) #(score, avg_loss)

def train(model_handler, num_epochs, early_stopping_patience=0, verbose=True, vld_data=None, tst_data=None, train_step_fn=None):
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
    # last_vld_loss = float("inf")
    greater_loss_epochs = 0

    if train_step_fn is None:
        train_step_fn = model_handler.train_step

    for epoch in range(num_epochs):
        train_step_fn()
        # print training loss 
        print("Total training loss: {}".format(model_handler.loss))

        # print training (& vld) scores
        # if verbose:
            # eval model on training data
            # trn_scores, trn_loss = eval_helper(
            #     model_handler,
            #     data_name='TRAIN'
            # )
            # trn_scores_dict[epoch] = copy.deepcopy(trn_scores)
            # print("Training loss: {}".format(trn_loss))

            # update best model checkpoint
        
        if vld_data is not None and early_stopping_patience > 0:
            vld_scores, vld_loss, vld_y_pred = eval_helper(
                model_handler,
                data_name='VALIDATION',
                data=vld_data
            )
            vld_scores_dict[epoch] = copy.deepcopy(vld_scores)
            # print vld loss
            if verbose:
                print("Avg Validation loss: {}".format(vld_loss))

            #check if is best vld loss to save best model
            if vld_loss < best_vld_loss:
                best_vld_loss = vld_loss
                greater_loss_epochs = 0
                model_handler.save_best()

            # check if the current vld loss is greater than the last loss and
            # break the training loop if its over the early stopping patience
            greater_loss_epochs += vld_loss >= best_vld_loss
            if greater_loss_epochs > early_stopping_patience:
                break
            
            if model_handler.has_scheduler:
                model_handler.scheduler.step(vld_loss)
        else:
            model_handler.save_best()
            
        if tst_data is not None and verbose:
            tst_scores, tst_loss, tgt_y_pred = eval_helper(
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

def train_AAD(model_handler, num_epochs, early_stopping_patience=0, verbose=True, vld_data=None, tst_data=None):
    '''
    Trains the given AAD model using the given data for the specified
    number of epochs. Prints training loss and evaluation. Saves at
    most 1 checkpoint plus a final one.
    :param model_handler: a holder with a model and data to be trained.
                            Assuming the model is a pytorch model.
    :param num_epochs: the number of epochs to train the model for.
    :param verbose: whether or not to print train results while training.
                    Default (True): do print intermediate results.
    '''
    print("*"*25)
    print("Pre-Training")
    print("*"*25)
    train(
        model_handler=model_handler,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
        vld_data=vld_data,
        tst_data=tst_data,
        train_step_fn=model_handler.pretrain_step,
    )

    print("*"*25)
    print("Adapting")
    print("*"*25)
    train(
        model_handler=model_handler,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
        vld_data=vld_data,
        tst_data=tst_data,
        train_step_fn=model_handler.adapt_step,
    )

def train_JointCL(model_handler, num_epochs, early_stopping_patience=0, verbose=True, vld_data=None, tst_data=None, train_step_fn=None):
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
    # last_vld_loss = float("inf")
    greater_loss_epochs = 0

    if train_step_fn is None:
        train_step_fn = model_handler.train_step

    for epoch in range(num_epochs):
        train_step_fn()
        # print training loss 
        print("Total training loss: {}".format(model_handler.loss))

        # print training (& vld) scores
        # if verbose:
            # eval model on training data
            # trn_scores, trn_loss = eval_helper(
            #     model_handler,
            #     data_name='TRAIN'
            # )
            # trn_scores_dict[epoch] = copy.deepcopy(trn_scores)
            # print("Training loss: {}".format(trn_loss))

            # update best model checkpoint
        if vld_data is not None and early_stopping_patience > 0:
            vld_scores, vld_loss, vld_y_pred = eval_helper(
                model_handler,
                data_name='VALIDATION',
                data=vld_data
            )
            vld_scores_dict[epoch] = copy.deepcopy(vld_scores)
            # print vld loss
            if verbose:
                print("Avg Validation loss: {}".format(vld_loss))

            #check if is best vld loss to save best model
            if vld_loss < best_vld_loss:
                best_vld_loss = vld_loss
                greater_loss_epochs = 0
                model_handler.save_best()

            # check if the current vld loss is greater than the last loss and
            # break the training loop if its over the early stopping patience
            greater_loss_epochs += vld_loss >= best_vld_loss
            if greater_loss_epochs > early_stopping_patience:
                break
            
            if model_handler.has_scheduler:
                model_handler.scheduler.step(vld_loss)
        else:
            model_handler.save_best()
            
        if tst_data is not None and verbose:
            tst_scores, tst_loss, tgt_y_pred = eval_helper(
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

def save_predictions(model_handler, dev_data, dev_dataloader, out_name, config, is_test=False, is_valid=False, dev_results=None):#, correct_preds=False):
    # trn_results = model_handler.predict()
    # trn_preds = trn_results[0]
    
    if dev_results is None:
        dev_results = model_handler.predict(data=dev_dataloader)#, correct_preds=correct_preds)
        dev_results = dev_results[0]
        # dev_preds = dev_results[0]
    
    dev_preds = []
    dev_proba = []

    if hasattr(model_handler, "is_llm") and model_handler.is_llm:
        dev_proba = dev_results
        dev_preds = np.floor(dev_results*model_handler.num_labels)
    
    elif int(config['n_output_classes']) == 2:
        for pred_val in dev_results:
            dev_proba.append(pred_val)

            int_pred = int(pred_val > 0.5)
            vec_pred = (int_pred,)

            dev_preds.append(dev_data.convert_vec_to_lbl(vec_pred))
    
    else:
        base_vec = [0 for i in range(int(config['n_output_classes']))]
        
        for pred_val in dev_results:
            dev_proba.append(pred_val)
            
            int_pred = pred_val.argmax()
            vec_pred = np.array(base_vec)*0
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
    if out_name is not None:
        if "/" in out_name:
            out_folder = "/".join(out_name.split("/")[:-1])
            os.makedirs(out_folder, exist_ok=True)
        
        predict_helper(dev_preds, dev_proba, dev_data, config).to_csv(out_name + '-{}.csv'.format(dev_name), index=False)
        print("saved to {}-{}.csv".format(out_name, dev_name))

def predict_helper(pred_lst, proba_lst, pred_data, config):
    out_data = []
    cols = [
        pred_data.text_col,
        pred_data.topic_col,
        pred_data.label_col,
    ]
    for i in pred_data.df.index:
        row = pred_data.df.iloc[i]
        temp = [row[c] for c in cols]
        temp.append(pred_lst[i])
        temp.append(proba_lst[i])
        out_data.append(temp)
    cols += [pred_data.label_col+'_pred', pred_data.label_col+'_proba']
    return pd.DataFrame(out_data, columns=cols)

def llm_collate_fn(batch):
    collated_dict = {}

    for instance in batch:
        for k, v in instance.items():
            # if isinstance(v, str): 
            collated_dict[k] = collated_dict.get(k, []) + [v]
            # else:
            #     collated_dict[k] = collated_dict.get(k, []) + [torch.Tensor(v)]
    
    return collated_dict

def main(args):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    ####################
    # load config file #
    ####################
    config = load_config_file(args['config_file'])
    is_llm = config.get("model_type","").lower() in ["llama_cpp", "hf_llm", "hf_api"]
    is_hfllm = config.get("model_type","").lower() == "hf_llm"
    needs_few_shot_examples = int(config.get("needs_few_shot_examples", "0"))

    #############
    # LOAD DATA #
    #############
    # load training data
    batch_size = args.get("batch_size") or (int(config['batch_size']) if not is_llm else 1)
    train_fn = train
    train_data = None
    if args['trn_data'] is not None:
        train_data = load_data(config, args, data_key="trn")
        train_n_batches = math.ceil(len(train_data) / batch_size)
        print(f"# of Training instances: {len(train_data)}. Batch Size={batch_size}. # of batches: {train_n_batches}")

        trn_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=args["drop_last_batch"],
            collate_fn = llm_collate_fn if is_hfllm else None,
            # collate_fn = None,
        )
    else:
        if args["mode"] == "train":
            raise "Please use a dataset to train the model."
        trn_dataloader = None

    # load vld data if specified
    if args['vld_data'] is not None:
        vld_data = load_data(config, args, data_key="vld", trn_data=train_data)
        vld_n_batches = math.ceil(len(vld_data) / batch_size)
        print(f"# of Validation instances: {len(vld_data)}. Batch Size={batch_size}. # of batches: {vld_n_batches}")

        vld_dataloader = torch.utils.data.DataLoader(
            vld_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=args["drop_last_batch"],
            collate_fn = llm_collate_fn if is_hfllm else None,
        )
    else:
        vld_dataloader = None

    # load tst data if specified
    if args['tst_data'] is not None:
        tst_data = load_data(config, args, data_key="tst", trn_data=train_data)
        tst_n_batches = math.ceil(len(tst_data) / batch_size)
        print(f"# of Test instances: {len(tst_data)}. Batch Size={batch_size}. # of batches: {tst_n_batches}")

        tst_dataloader = torch.utils.data.DataLoader(
            tst_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=args["drop_last_batch"],
            collate_fn = llm_collate_fn if is_hfllm else None,
        )
    else:
        tst_dataloader = None

    lr = float(config.get('learning_rate', '0.001'))
    nl = int(config.get("n_output_classes", "3"))

    # RUN
    print("Using cuda?: {}".format(use_cuda))
    hf_model_params = get_config_dict_prefix(
        config_dict=config,
        prefix="hf_model_"
    )

    if "BiLSTMAttn" in config["name"]:
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
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
            loader_params=hf_model_params,
        )

        topic_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
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
            loader_params=hf_model_params,
        )

        topic_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
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

    elif "ensemble" in config["name"].lower():
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
        )

        base_params_dict = {
            "text_input_dim": text_input_layer.dim,
        }
        if config["gating_model_type"].lower() != "bilstmattn":
            topic_input_layer = input_models.BertLayer(
                use_cuda=use_cuda,
                pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
                layers=config.get("bert_layers", "-1"),
                layers_agg_type=config.get("bert_layers_agg", "concat"),
            )
            base_params_dict["topic_input_dim"] = topic_input_layer.dim
        else:
            topic_input_layer = None
        
        loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
        if nl < 3:
            loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)#ignore_index=nl)

        # n_clfs = 2#int(config.get("n_experts", 3))
        # expert_params = [copy.deepcopy(base_params_dict) for _ in range(n_experts)]
        gating_params = copy.deepcopy(base_params_dict)
        
        for key, value in config.items():
            if key.startswith("gating_"):
                gating_params[key.replace("gating_", "")] = value
            # elif key.startswith("expert_"):
            #     expert_number = int(key.split("_")[1])
            #     expert_params[expert_number-1][key.replace(f"expert_{expert_number}_", "")] = value

        model = models.EnsembleModel(
            gating_params=gating_params,
            num_labels=nl,
            use_cuda=use_cuda,
        )

        o = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=o,
            patience=2,
        )

        kwargs = {
            'model': model,
            'text_input_model': text_input_layer,
            'topic_input_model': topic_input_layer,
            'dataloader': trn_dataloader,
            'name': config['name'] + args['name'],
            'loss_function': loss_fn,
            'optimizer': o,
            'scheduler': scheduler,
            'is_joint_text_topic':config.get("is_joint"),
        }

        model_handler = model_utils.TorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )

    elif 'CrossNet' in config['name']:
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
        )

        topic_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
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

    elif 'TOAD' in config['name']:
        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
        )

        topic_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
        )

        model = models.TOAD(
            hidden_dim=int(config['lstm_hidden_dim']),
            text_input_dim=text_input_layer.dim,
            topic_input_dim=topic_input_layer.dim,
            stance_dim=int(config['stance_classifier_dimension']),
            topic_dim=int(config["topic_classifier_dimension"]),
            num_topics=train_data.get_num_topics(),
            proj_layer_dim=int(config.get("proj_layer_dim", 128)),
            num_layers=int(config.get("lstm_layers","1")),
            num_labels=nl,
            drop_prob=float(config.get('dropout', "0")),
            use_cuda=use_cuda,
        )
        
        loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
        loss_fn = lf.TOADLoss(
            trans_dim=2*int(config['lstm_hidden_dim']),
            trans_param=float(config['transformation_loss_w']),
            num_no_adv=float(config['num_no_adv_loss']),
            tot_epochs=int(config['epochs']),
            rho_adv=('rho_adv' in config),
            gamma=float(config.get('gamma', 10.0)),
            semi_sup=bool(config.get('semi_sup', False)),
            use_cuda=use_cuda,
            n_outputs=nl
        )

        optim_fn = torch.optim.Adam
        opt_main_add_params = {
            "lr": lr,
            "weight_decay": float(config.get('l2_main', '0')),
        }
        opt_adv_add_params = {
            "lr": lr,
            "weight_decay": float(config.get('l2_adv', '0')),
        }
        if config.get('optimizer') == 'sgd':
            optim_fn = torch.optim.SGD
            sgd_dict = {
                "momentum": 0.9,
                "nesterov": True
            }
            opt_main_add_params.update(sgd_dict)
            opt_adv_add_params.update(sgd_dict)
            
        opt_main = optim_fn(
            itertools.chain(
                model.enc.parameters(),
                model.text_recon_layer.parameters(),
                model.topic_recon_layer.parameters(),
                model.trans_layer.parameters(),
                model.stance_classifier.parameters()
            ),
            **opt_main_add_params
        )
        
        opt_adv = optim_fn(
            model.topic_classifier.parameters(),
            **opt_adv_add_params
        )

        kwargs = {
            'model': model,
            'text_input_model': text_input_layer,
            'topic_input_model': topic_input_layer,
            'dataloader': trn_dataloader,
            'name': config['name'] + args['name'],
            'loss_function': loss_fn,
            'optimizer': opt_main,
            'adv_optimizer': opt_adv,
            'tot_epochs': int(config['epochs']),
            'initial_lr': lr,
            'alpha': float(config.get('alpha', 10.0)),
            'beta': float(config.get('beta', 0.75)),
            'num_constant_lr': float(config['num_constant_lr']),
            'is_joint_text_topic':config.get("is_joint"),
        }

        model_handler = model_utils.TOADTorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )
    
    elif 'AAD' in config['name']:
        if args["mode"] == "train":
            train_fn = train_AAD
            tgt_train_data = load_data(config, args, data_key="tgt_trn")
            tgt_train_n_batches = math.ceil(len(tgt_train_data) / batch_size)
            print(f"# of Training instances: {len(tgt_train_data)}. Batch Size={batch_size}. # of batches: {tgt_train_n_batches}")

            tgt_trn_dataloader = torch.utils.data.DataLoader(
                tgt_train_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=args["drop_last_batch"]
            )
        else:
            tgt_trn_dataloader = None

        src_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
        )

        tgt_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased"),
            layers=config.get("bert_layers", "-1"),
            layers_agg_type=config.get("bert_layers_agg", "concat"),
            loader_params=hf_model_params,
        )

        model = models.AAD(
            text_input_dim=src_input_layer.dim,
            src_encoder=src_input_layer,
            tgt_encoder=tgt_input_layer,
            discriminator_dim=int(config['discriminator_dim']),
            num_labels=nl,
            drop_prob=float(config.get('dropout', "0")),
            use_cuda=use_cuda,
        )

        loss_reduction = "none" if bool(int(config.get("sample_weights", "0"))) else "mean"
        if nl < 3:
            src_encoder_loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
            tgt_encoder_loss_fn = torch.nn.BCELoss(reduction=loss_reduction)
        else:
            src_encoder_loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)
            tgt_encoder_loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_reduction)
        disc_loss = torch.nn.BCELoss()
        
        src_encoder_opt = torch.optim.AdamW(
            list(src_input_layer.parameters()) + list(model.classifier.parameters()),
            lr = float(config.get('learning_rate', '2e-5')),
            eps = 1e-8
        )
        tgt_encoder_opt = torch.optim.Adam(
            tgt_input_layer.parameters(),
            lr = float(config.get('discriminator_learning_rate', '1e-5')),
        )
        discriminator_opt = torch.optim.Adam(
            model.discriminator.parameters(),
            lr = float(config.get('discriminator_learning_rate', '1e-5'))
        )
        kwargs = {
            'model': model,
            'tgt_dataloader': tgt_trn_dataloader,
            'text_input_model': src_input_layer,
            'tgt_text_input_model': tgt_input_layer,
            'dataloader': trn_dataloader,
            'name': config['name'] + args['name'],
            'loss_function': src_encoder_loss_fn,
            'optimizer': src_encoder_opt,
            'discriminator_loss_fn': disc_loss,
            'discriminator_clip_value': float(config.get('discriminator_clip_value', '0.01')),
            'discriminator_optimizer': discriminator_opt,
            'tgt_encoder_optimizer': tgt_encoder_opt,
            'tgt_encoder_temperature': int(config.get('tgt_encoder_temperature', '20')),
            'tgt_encoder_loss_fn': tgt_encoder_loss_fn,
            'tgt_loss_alpha': float(config.get('tgt_loss_alpha', '1.0')),
            'tgt_loss_beta': float(config.get('tgt_loss_beta', '1.0')),
            'max_grad_norm': float(config.get('max_grad_norm', '1.0')),
        }

        model_handler = model_utils.AADTorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )

    elif 'JointCL' in config['name']:
        # needs to save the clusters to be able to predict with the trained model
        pretrained_model_name=config.get("bert_pretrained_model", "bert-base-uncased")
        layers=config.get("bert_layers", "-1")
        layers_agg_type=config.get("bert_layers_agg", "concat")

        device = torch.device('cuda' if use_cuda else 'cpu')
        temperature = float(config.get("temperature", "0.07"))
        dp = float(config.get("dp", "0.1"))
        dropout = float(config.get("dropout", "0.1"))
        gnn_dims = config.get("gnn_dims", "192,192")
        att_heads = config.get("att_heads", "4,4")
        alpha = config.get("alpha", "0.8")
        beta = config.get("beta", "1.2")

        cluster_times = int(config.get("cluster_times", "1"))
        prototype_loss_weight = float(config.get("prototype_loss_weight", "0.2"))
        stance_loss_weight = float(config.get("stance_loss_weight", "1.0"))

        if args["mode"] == "train":
            train_loader_prototype = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=args["drop_last_batch"],
            )
        else:
            train_loader_prototype = None

        text_input_layer = input_models.BertLayer(
            use_cuda=use_cuda,
            pretrained_model_name=pretrained_model_name,
            layers=layers,
            layers_agg_type=layers_agg_type,
            loader_params=hf_model_params,
        )
        
        stance_loss_fn = JointCL_loss.Stance_loss(temperature).to(device)
        target_loss_fn = JointCL_loss.Stance_loss(temperature).to(device)
        logits_loss_fn = JointCL_loss.TraditionCriterion(batch_size=batch_size, num_labels=nl)

        model = models.JointCL(
            att_heads=att_heads,
            bert_dim=text_input_layer.dim,
            bert_layer=text_input_layer,
            dp=dp,
            dropout=dropout,
            gnn_dims=gnn_dims,
            num_labels=nl,
            use_cuda=use_cuda,
        )

        params2optimize = ([p for p in model.parameters()] + [p for p in target_loss_fn.parameters()])
        opt = torch.optim.Adam(
            params2optimize,
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
            'loss_function': None,
            'optimizer': opt,
            # 'scheduler': scheduler,
            'is_joint_text_topic':config.get("is_joint"),
            'bert_dim': text_input_layer.dim,
            'cluster_times': cluster_times,
            'device': device,
            'logits_loss_fn': logits_loss_fn,
            'prototype_loss_weight': prototype_loss_weight,
            'stance_loss_fn': stance_loss_fn,
            'stance_loss_weight': stance_loss_weight,
            'target_loss_fn': target_loss_fn,
            'temperature': temperature,
            'train_loader_prototype': train_loader_prototype,
        }

        model_handler = model_utils.JointCLTorchModelHandler(
            checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )

    elif is_llm:
        if 'llama_cpp' in config['model_type']:
            
            llama_cpp_params = {
                "model_path": config["pretrained_model_name"],
            }
            for k,v in config.items():
                if k.startswith("llama_cpp_"):
                    llama_cpp_params[k.replace("llama_cpp_", "")] = v
            
            model = llms.LlamaCpp_Model(
                params=llama_cpp_params,
                num_labels=nl
            )

        elif "hf_llm" in config['model_type']:
            bitsandbytesconfig_params = get_config_dict_prefix(
                config_dict=config,
                prefix="bitsandbytesconfig_"
            )
            
            model = llms.HF_Llama_Model(
                model=config.get("pretrained_model_name", "bigscience/bloom-1b7"),
                hf_model_params=hf_model_params,
                bitsandbytesconfig_params=bitsandbytesconfig_params,
                num_labels=nl,
            )
        
        elif "hf_api" in config['model_type']:
            model_params = get_config_dict_prefix(
                config_dict=config,
                prefix="hf_api_"
            )
            
            model = llms.HF_API_Model(
                api_url=config.get("api_url", "https://api-inference.huggingface.co/models/bigscience/bloom"),
                auth_token=config.get("auth_token"),
                params=model_params,
                num_labels=nl,
            )
        
        llm_params = get_config_dict_prefix(config_dict=config, prefix="llm_")
        
        if "tst_data" in locals():
            dataset_ = tst_data
            dataloader_ = tst_dataloader
        elif "vld_data" in locals():
            dataset_ = vld_data
            dataloader_ = vld_dataloader
        elif "train_data" in locals():
            dataset_ = train_data
            dataloader_ = trn_dataloader

        kwargs = {
            "model": model,
            "text_input_model": None,
            "topic_input_model": None,
            "is_joint_text_topic": False,
            "dataloader": dataloader_,
            "name": config['name'] + args['name'],
            "loss_function": None,
            "optimizer": None,
            #specific params
            "model_params": llm_params,
            "dataset": dataset_,
            "output_format": config["output_format"],
            "save_every_n_batches": config.get("save_every_n_batches", "2"),
            "output_err_default": config.get("output_err_default", "0"),
            "output_class_order": config.get("output_class_order", None),
            "is_hfllm": is_hfllm
            # "output_parser": [use specifc function if needed]
        }

        if "output_max_score" in config:
            kwargs["output_max_score"] = float(config["output_max_score"])

        model_handler = model_utils.LLMTorchModelHandler(
            checkpoint_path = args.get('out_path') or config.get('ckp_path', 'data/checkpoints/'),
            use_cuda=use_cuda,
            **kwargs
        )

    if args["mode"] == 'train':
        # Train model
        start_time = time.time()
        train_fn(
            model_handler=model_handler,
            num_epochs=int(config['epochs']),
            early_stopping_patience=args.get("early_stop", 0),
            verbose=True,
            vld_data=vld_dataloader,
            tst_data=tst_dataloader,
        )
        print(f"[{config['name']}] total runtime: {(time.time() - start_time)/60:.2f} minutes")

    trn_y_pred = None
    if args["trn_results"] != "":
        with open(args["trn_results"], mode="r", encoding="utf-8") as f_:
            trn_y_pred = json.load(f_)["pred"]

    vld_y_pred = None
    if args["vld_results"] != "":
        with open(args["vld_results"], mode="r", encoding="utf-8") as f_:
            vld_y_pred = json.load(f_)["pred"]

    tst_y_pred = None
    if args["tst_results"] != "":
        with open(args["tst_results"], mode="r", encoding="utf-8") as f_:
            tst_y_pred = json.load(f_)["pred"]
    
    if 'eval' in args["mode"]:
        # Evaluate saved model
        if not is_llm:
            model_handler.load(filename=args["saved_model_file_name"])
        
        if trn_dataloader is not None and not needs_few_shot_examples:
            trn_scores, trn_loss, trn_y_pred = eval_helper(
                model_handler,
                data_name='TRAIN',
                data=trn_dataloader,
                y_pred=trn_y_pred,
            )
        
        if vld_dataloader is not None:
            vld_scores, vld_loss, vld_y_pred = eval_helper(
                model_handler,
                data_name='VALIDATION',
                data=vld_dataloader,
                y_pred=vld_y_pred,
            )

        if tst_dataloader is not None:
            tst_scores, tst_loss, tst_y_pred = eval_helper(
                model_handler,
                data_name='TEST',
                data=tst_dataloader,
                y_pred=tst_y_pred,
            )

    if "pred" in args["mode"]:
        out_path = args.get("out_path", "./pred")
        print("Output Path:", out_path)
        # laod saved model and save the predictions
        if not is_llm:
            model_handler.load(filename=args["saved_model_file_name"])
        
        if trn_dataloader is not None and not needs_few_shot_examples:
            save_predictions(
                model_handler=model_handler,
                dev_data=train_data,
                dev_dataloader=trn_dataloader,
                out_name=out_path,
                config=config,
                is_test=False,#('test' in args["trn_data"] or 'tst' in args["trn_data"]),
                is_valid=False,#('valid' in args["trn_data"] or 'vld' in args["trn_data"]),
                dev_results=trn_y_pred,
            )
            
        if vld_dataloader is not None:
            save_predictions(
                model_handler=model_handler,
                dev_data=vld_data,
                dev_dataloader=vld_dataloader,
                out_name=out_path,
                config=config,
                is_test=False,#('test' in args["vld_data"] or 'tst' in args["vld_data"]),
                is_valid=True,#('valid' in args["vld_data"] or 'vld' in args["vld_data"]),
                dev_results=vld_y_pred,
            )
        
        if tst_dataloader is not None:
            save_predictions(
                model_handler=model_handler,
                dev_data=tst_data,
                dev_dataloader=tst_dataloader,
                out_name=out_path,
                config=config,
                is_test=True,#('test' in args["tst_data"] or 'tst' in args["tst_data"]),
                is_valid=False,#('valid' in args["tst_data"] or 'vld' in args["tst_data"]),
                dev_results=tst_y_pred,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest="mode", help='What to do', required=True)
    parser.add_argument('-c', '--config_file', dest="config_file", help='Name of the cofig data file', required=False)
    parser.add_argument('-t', '--trn_data', dest="trn_data", help='Name of the training data file', required=False)
    parser.add_argument('-g', '--tgt_trn_data', dest="tgt_trn_data", help='Name of the training data file containing only the target domain (exclusive for AAD)', required=False)
    parser.add_argument('-v', '--vld_data', dest="vld_data", help='Name of the validation data file', default=None, required=False)
    parser.add_argument('-p', '--tst_data', dest="tst_data", help='Name of the test data file', default=None, required=False)
    parser.add_argument('-n', '--name', dest="name", help='something to add to the saved model name', required=False, default='')
    parser.add_argument('-e', '--early_stop', dest="early_stop", help='Whether to do early stopping or not', required=False, type=int, default=False)
    # parser.add_argument('-w', '--num_warm', dest="num_warm", help='Number of warm-up epochs', required=False, type=int, default=0)
    # parser.add_argument('-k', '--score_key', dest="score_key", help='Score to use for optimization', required=False, default='f_macro')
    parser.add_argument('-s', '--save_ckp', dest="save_ckp", help='Whether to save checkpoints', required=False, default=0, type=int)
    parser.add_argument('-f', '--saved_model_file_name', dest="saved_model_file_name", required=False, default=None)
    parser.add_argument('-o', '--out_path', dest="out_path", help='Ouput file name', default=None)
    parser.add_argument('-d', '--drop_last_batch', type=bool, dest="drop_last_batch", help="Whether to drop the last batch or not", default=False)
    parser.add_argument('-a', '--train_data_sample', type=float, dest="train_data_sample", help="Value to sample the train data.", default=1.0)
    parser.add_argument('-r', '--random_state', type=int, dest="random_state", help="Random state seed to sample the train data.", default=123)
    parser.add_argument('-k', '--skip_rows_trn', type=int, dest="skip_rows_trn", help="Skip rows in train data.", default=0)
    parser.add_argument('-j', '--skip_rows_vld', type=int, dest="skip_rows_vld", help="Skip rows in validation data.", default=0)
    parser.add_argument('-l', '--skip_rows_tst', type=int, dest="skip_rows_tst", help="Skip rows in test data.", default=0)
    parser.add_argument('-u', '--tst_results', type=str, dest="tst_results", help="Path to file containing the test results. (predictions)", default="")
    parser.add_argument('-w', '--vld_results', type=str, dest="vld_results", help="Path to file containing the validation results. (predictions)", default="")
    parser.add_argument('-x', '--trn_results', type=str, dest="trn_results", help="Path to file containing the train results. (predictions)", default="")
    parser.add_argument('-bt1', '--ensemble_clf1_pred_trn', type=str, dest="ensemble_clf1_pred_trn", help="Path to file containing the 1st classifier predictions on train data.", default=None)
    parser.add_argument('-bt2', '--ensemble_clf2_pred_trn', type=str, dest="ensemble_clf2_pred_trn", help="Path to file containing the 2nd classifier predictions on train data.", default=None)
    parser.add_argument('-bv1', '--ensemble_clf1_pred_vld', type=str, dest="ensemble_clf1_pred_vld", help="Path to file containing the 1st classifier predictions on validation data.", default=None)
    parser.add_argument('-bv2', '--ensemble_clf2_pred_vld', type=str, dest="ensemble_clf2_pred_vld", help="Path to file containing the 2nd classifier predictions on validation data.", default=None)
    parser.add_argument('-bp1', '--ensemble_clf1_pred_tst', type=str, dest="ensemble_clf1_pred_tst", help="Path to file containing the 1st classifier predictions on test data.", default=None)
    parser.add_argument('-bp2', '--ensemble_clf2_pred_tst', type=str, dest="ensemble_clf2_pred_tst", help="Path to file containing the 2nd classifier predictions on test data.", default=None)
    parser.add_argument('-tt', '--trn_text_col', type=str, dest="trn_text_col", help="Override the name of text column for training dataset.", required=False)
    parser.add_argument('-tg', '--trn_topic_col', type=str, dest="trn_topic_col", help="Override the name of topic column for training dataset.", required=False)
    parser.add_argument('-tl', '--trn_label_col', type=str, dest="trn_label_col", help="Override the name of label column for training dataset.", required=False)
    parser.add_argument('-vt', '--vld_text_col', type=str, dest="vld_text_col", help="Override the name of text column for validation dataset..", required=False)
    parser.add_argument('-vg', '--vld_topic_col', type=str, dest="vld_topic_col", help="Override the name of topic column for validation dataset..", required=False)
    parser.add_argument('-vl', '--vld_label_col', type=str, dest="vld_label_col", help="Override the name of label column for validation dataset..", required=False)
    parser.add_argument('-pt', '--tst_text_col', type=str, dest="tst_text_col", help="Override the name of text column for test dataset.", required=False)
    parser.add_argument('-pg', '--tst_topic_col', type=str, dest="tst_topic_col", help="Override the name of topic column for test dataset.", required=False)
    parser.add_argument('-pl', '--tst_label_col', type=str, dest="tst_label_col", help="Override the name of label column for test dataset.", required=False)
    parser.add_argument('-bs', '--batch_size', type=int, dest="batch_size", help="Override the batch size for loading all datasets.", required=False)
    args = vars(parser.parse_args())

    main(args)

# running

## LOCAL
# train BiCondBertLstm
# python train_model.py -m train -c ../../config/BiCondBertLstm_example.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -n bo -e 5 -s 1
# python train_model.py -m train -c ../../config/BiCondBertLstm_example.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1


# train BertBiLSTMJointAttn
# python train_model.py -m train -c ../../config/Bert_BiLstmJointAttn_example.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -p ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1
# python train_model.py -m train -c ../../config/Bert_BiLstmJointAttn_example.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -p ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 5 -s 1

# train BertBiLSTMAttn (Simple Domain)
# python train_model.py -m train -c ../../config/Bert_BiLstmAttn_example.txt -t ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 5 -s 1

# predict BertBiLSTMAttn (Simple Domain)
# python train_model.py -m predict -c ../../config/Bert_BiLstmAttn_example.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -f ../../checkpoints/ustancebr/V0/ckp-BertBiLSTMAttn_ustancebr_bo-BEST.tar -o ../../out/ustancebr/pred/BertBiLSTMAttn_bo_BEST_v0
# -t ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv

# eval BertBiLSTMAttn (Simple Domain)
# python train_model.py -m eval -c ../../config/simple_domain/Bert_BiLSTMAttn_v1.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -f ../../checkpoints/ustancebr/simple_domain/V1/ckp-BertBiLSTMAttn_ustancebr_bo-BEST.tar -o ../../out/ustancebr/eval/BertBiLSTMAttn_bo_BEST_v1 -t ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv

# train Ensemble (Hold1TopicOut)
# python train_model.py -m train -c ../../config/example/Bert_Ensemble_example.txt -t ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_train.csv -v ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_valid.csv -p ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_test.csv -n bo -e 5 -s 1 -bt1 ../../data/ustancebr/v2/teste_pqno_h1to/clf_1_pred_pqno_bo_train.csv -bt2 ../../data/ustancebr/v2/teste_pqno_h1to/clf_2_pred_pqno_bo_train.csv -bv1 ../../data/ustancebr/v2/teste_pqno_h1to/clf_1_pred_pqno_bo_valid.csv -bv2 ../../data/ustancebr/v2/teste_pqno_h1to/clf_2_pred_pqno_bo_valid.csv -bp1 ../../data/ustancebr/v2/teste_pqno_h1to/clf_1_pred_pqno_bo_test.csv -bp2 ../../data/ustancebr/v2/teste_pqno_h1to/clf_2_pred_pqno_bo_test.csv

# train CrossNet (Simple Domain)
# python train_model.py -m train -c ../../config/Bert_CrossNet_example.txt -t ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 2 -s 1

# train TOAD (Hold1TopicOut)
# python train_model.py -m train -c ../../config/Bert_TOAD_example.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -p ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 2 -s 1
# python train_model.py -m train -c ../../config/Bert_TOAD_example.txt -t ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_train.csv -v ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_test.csv -p ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_test.csv -n bo -e 2

# train TOAD (SEMEVAL Hold1TopicOut)
# python train_model.py -m train -c ../../config/Bert_TOAD_example_Semeval.txt -t ../../data/semeval/hold1topic_out/final_dt_train.csv -v ../../data/semeval/hold1topic_out/final_dt_valid.csv -p ../../data/semeval/hold1topic_out/final_dt_test.csv -n dt -e 2 -s 1

# train TOAD (SEMEVAL indomain)
# python train_model.py -m train -c ../../config/Bert_TOAD_example_Semeval.txt -t ../../data/semeval/indomain/final_hc_train.csv -v ../../data/semeval/indomain/final_hc_valid.csv -p ../../data/semeval/indomain/final_hc_test.csv -n hc -e 2 -s 1

# train AAD (Simple Domain)
# python train_model.py -m train -c ../../config/Bert_AAD_example.txt -t ../../data/ustancebr/v2/simple_domain/final_lu_train.csv -g ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 2 -s 1
# python train_model.py -m train -c ../../config/Bert_AAD_example.txt -t ../../data/ustancebr/v2/teste_pqno/pqno_lu_train.csv -g ../../data/ustancebr/v2/teste_pqno/pqno_bo_train.csv -v ../../data/ustancebr/v2/teste_pqno/pqno_bo_train.csv -p ../../data/ustancebr/v2/teste_pqno/pqno_bo_test.csv -n bo -e 2 -s 1

# train JointCL (ustancebr Simple Domain)
# python train_model.py -m train -c ../../config/Bert_JointCL_example.txt -t ../../data/ustancebr/v2/simple_domain/final_bo_train.csv -v ../../data/ustancebr/v2/simple_domain/final_bo_valid.csv -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -n bo -e 2 -s 1
# train JointCL (ustancebr Hold1TopicOut)
# python train_model.py -m train -c ../../config/Bert_JointCL_example.txt -t ../../data/ustancebr/v2/hold1topic_out/final_bo_train.csv -v ../../data/ustancebr/v2/hold1topic_out/final_bo_valid.csv -p ../../data/ustancebr/v2/hold1topic_out/final_bo_test.csv -n bo -e 2 -s 1
# train JointCL (ustancebr PQNO Hold1TopicOut)
# python train_model.py -m train -c ../../config/Bert_JointCL_example.txt -t ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_train.csv -v ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_valid.csv -p ../../data/ustancebr/v2/teste_pqno_h1to/pqno_bo_test.csv -n bo -e 2

# train JointCL (Semeval indomain)
# python train_model.py -m train -c ../../config/Bert_JointCL_example_semeval.txt -t ../../data/semeval/indomain/final_dt_train.csv -v ../../data/semeval/indomain/final_dt_valid.csv -p ../../data/semeval/indomain/final_dt_test.csv -n bo -e 2 -s 1
# train JointCL (Semeval Hold1TopicOut)
# python train_model.py -m train -c ../../config/Bert_JointCL_example_semeval.txt -t ../../data/semeval/hold1topic_out/final_dt_train.csv -v ../../data/semeval/hold1topic_out/final_dt_valid.csv -p ../../data/semeval/hold1topic_out/final_dt_test.csv -n bo -e 2 -s 1

# predict llama_4bit
# python train_model.py -m predict -c ../../config/Llama_4bit_example.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -o ../../out/ustancebr/pred/zero_shot/Llama_4bit_bo_v0
# python train_model.py -m predict -c ../../config/Llama_4bit_example.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -o ../../out/ustancebr/pred/zero_shot/Llama_4bit_bo_v0 -l 2300
# python train_model.py -m predict -c ../../config/Llama_4bit_example.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -o ../../out/ustancebr/pred/zero_shot/Llama_4bit_bo_v0 -u ../../out/ustancebr/pred/zero_shot/Llama_4bit_bo_v0/llama_cpp_pred_checkpoints/Llama_4bit_ustancebr_full.ckp
# python train_model.py -m eval  -c ../../config/Llama_4bit_example.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -o ../../out/ustancebr/pred/zero_shot/Llama_4bit_bo_v0 -u ../../out/ustancebr/pred/zero_shot/Llama_4bit_bo_v0/llama_cpp_pred_checkpoints/Llama_4bit_ustancebr_full.ckp

# predict llama_4bit (Few-Shot) - SEMEVAL
# python train_model.py -m predict -c ../../config/semeval/few_shot/Llama_4bit_v0.txt -t ../../data/semeval/simple_domain/final_hc_train.csv -p ../../data/semeval/simple_domain/final_hc_test.csv -o ../../out/semeval/pred/few_shot/Llama_4bit_hc_v0 -a 10
# python train_model.py -m predict -c ../../config/semeval/few_shot/Llama_4bit_v0.txt -t ../../data/semeval/simple_domain/final_hc_train.csv -p ../../data/semeval/simple_domain/final_hc_test.csv -o ../../out/semeval/pred/few_shot/Llama_4bit_hc_v0 -u ../../out/semeval/pred/few_shot/Llama_4bit_hc_v0/llama_cpp_pred_checkpoints/Llama_4bit_semeval_full.ckp -a 10

# predict llama_8bit
# python train_model.py -m predict -c ../../config/Llama_8bit_example.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -o ../../out/ustancebr/pred/Llama_8bit_bo_v0

# predict llama3_4bit
# python train_model.py -m predict -c ../../config/Llama3_4bit_example.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -o ../../out/ustancebr/pred/Llama2_4bit_bo_v0
# python train_model.py -m predict -c ../../config/ustancebr/zero_shot/Llama3_4bit_v0.txt  -p ../../data/ustancebr/v2/simple_domain/final_bo_test.csv -o ../../out/ustancebr/pred/Llama3_4bit_bo_v0 -bs 12


# GOVBR
# python train_model.py -m train -c ../../config/govbr/simple_domain/BertBiLstmJointAttn_v12.txt -t ../../data/govbr/simple_domain/final_co_train.csv -v ../../data/govbr/simple_domain/final_co_valid.csv -p ../../data/ustancebr/v2/hold1topic_out/final_co_test.csv -n co -e 5 -s 1

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
