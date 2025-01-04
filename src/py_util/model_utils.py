import torch, datetime, time, math, numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import os
import json

class TorchModelHandler:
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be
    written in pytorch.
    '''
    def __init__(self, num_ckps=1, checkpoint_path="./data/checkpoints/", use_cuda=False, **params):

        self.model = params["model"]
        self.text_input_model = params["text_input_model"]
        self.topic_input_model = params.get("topic_input_model")
        self.is_joint_text_topic = bool(int(params.get("is_joint_text_topic", "0")))

        self.dataloader = params["dataloader"]
        self.num_labels = self.model.num_labels
        # self.output_dim = 1 if self.num_labels < 1 else self.num_labels
        self.output_dim = self.model.output_dim
        self.is_ensemble = self.model.is_ensemble if hasattr(self.model, "is_ensemble") else False

        self.name = params["name"]
        
        self.loss_function = params["loss_function"]
        self.optimizer = params["optimizer"]
        self.has_scheduler = "scheduler" in params
        if self.has_scheduler:
            self.scheduler = params["scheduler"]

        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.checkpoint_num = 0
        self.num_ckps = num_ckps
        # self.score_dict = dict()
        self.loss = 0
        self.epoch = 0

        self.use_cuda = use_cuda

    def save_best(self):
        '''
        Evaluates the model on data and then updates the best scores and saves the best model.
        '''
        self.save(num='BEST')

    def save(self, num=None):
        '''
        Saves the pytorch model in a checkpoint file.
        :param num: The number to associate with the checkpoint. By default uses
                    the internally tracked checkpoint number but this can be changed.
        '''
        if num is None:
            check_num = self.checkpoint_num
        else: check_num = num

        torch.save(
            obj = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss
            },
            f = f'{self.checkpoint_path}ckp-{self.name}-{check_num}.tar'
        )

        if num is None:
            self.checkpoint_num = (self.checkpoint_num + 1) % self.num_ckps

    def load(self, filename='data/checkpoints/ckp-[NAME]-FINAL.tar', use_cpu=False):
        '''
        Loads a saved pytorch model from a checkpoint file.
        :param filename: the name of the file to load from. By default uses
                        the final checkpoint for the model of this' name.
        '''
        filename = filename.replace('[NAME]', self.name)
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print(f"[{self.name}] epoch {self.epoch}")
        self.model.train()
        self.text_input_model.eval()
        if not self.is_joint_text_topic and self.topic_input_model:
            self.topic_input_model.eval()
        
        self.loss = 0
        # partial_loss = 0
        start_time = time.time()

        for batch_n, batch_data in tqdm(enumerate(self.dataloader)):
            #zero gradients before every optimizer step
            label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze(dim=-1)
            
            if self.num_labels <= 2:
                label_tensor = label_tensor.type(torch.FloatTensor)
            else:
                label_tensor = label_tensor.type(torch.LongTensor)
            
            if self.use_cuda:
                label_tensor = label_tensor.to("cuda")

            self.model.zero_grad()
            
            # get the embeddings for text and topic and creates a dict to pass as params to the model
            text_embeddings = self.text_input_model(**batch_data["text"])
            if self.is_joint_text_topic:
                model_inputs = {
                    "input":text_embeddings,
                    "input_length": batch_data["text"]["input_length"],
                }
            else:
                model_inputs = {
                    "text_embeddings": text_embeddings,
                    "text_length": batch_data["text"]["input_length"],
                }
                if self.topic_input_model:
                    topic_embeddings = self.topic_input_model(**batch_data["topic"])
                    model_inputs.update({
                        "topic_embeddings": topic_embeddings,
                        "topic_length": batch_data["topic"]["input_length"],
                    })
            
            if self.is_ensemble:
                clf_pred_label_1_tensor = torch.stack(batch_data["pred_label_1"]).T.type(torch.FloatTensor)
                if len(clf_pred_label_1_tensor.shape) > 1 and clf_pred_label_1_tensor.shape[-1] != 1:
                    clf_pred_label_1_tensor = clf_pred_label_1_tensor.argmax(dim=1).reshape(-1,1)
                clf_pred_label_1_tensor = clf_pred_label_1_tensor.squeeze(dim=-1)

                clf_pred_label_2_tensor = torch.stack(batch_data["pred_label_2"]).T.type(torch.FloatTensor)
                if len(clf_pred_label_2_tensor.shape) > 1 and clf_pred_label_2_tensor.shape[-1] != 1:
                    clf_pred_label_2_tensor = clf_pred_label_2_tensor.argmax(dim=1).reshape(-1,1)
                clf_pred_label_2_tensor = clf_pred_label_2_tensor.squeeze(dim=-1)

                model_inputs.update({
                    "clf1_pred": clf_pred_label_1_tensor,
                    "clf2_pred": clf_pred_label_2_tensor,
                })
            
            #apply the text and topic embeddings (and clf predictions if applicable) to the model
            y_pred = self.model(**model_inputs)
            if y_pred.squeeze(dim=-1).shape != torch.Size([]):
                y_pred = y_pred.squeeze(dim=-1)

            # calculate the loss, and backprogate it to update weights
            graph_loss = self.loss_function(y_pred, label_tensor)
            if "sample_weight" in batch_data:
                weight_lst = batch_data["sample_weight"]
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(graph_loss * weight_lst)
            
            graph_loss.backward()
            self.optimizer.step()

            #sum the loss
            # partial_loss += graph_loss.item()
            self.loss += graph_loss.item()

            #show the loss per batch once for each 1000 batches
            # if batch_n % 1000 == 999:
            #     last_loss = partial_loss / 1000 #loss per batch
            #     print(f"    batch {batch_n+1} loss: {last_loss}")
            #     partial_loss = 0

        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1

    def compute_scores(self, score_fn, true_labels, pred_labels, name, score_dict):
        '''
        Computes scores using the given scoring function of the given name. The scores
        are stored in the internal score dictionary.
        :param score_fn: the scoring function to use.
        :param true_labels: the true labels.
        :param pred_labels: the predicted labels.
        :param name: the name of this score function, to be used in storing the scores.
        :param score_dict: the dictionary used to store the scores.
        '''
        if hasattr(self, "is_llm") and self.is_llm:
            if self.output_dim > 1:
                # print("*"*20, "\ntrue_labels:", true_labels, "\n\n")
                # print("*"*20, "\npred_labels:", pred_labels, "\n\n")
                # print("*"*20, "\npred_labels*n_labels:", np.floor(pred_labels*self.num_labels), "\n\n")
                vals = score_fn(
                    true_labels,
                    np.floor(pred_labels*self.num_labels),
                    average=None,
                    labels=range(self.num_labels)
                )
            else:
                vals = score_fn(
                    true_labels,
                    (pred_labels>0.5)*1,
                    average=None,
                    labels=range(self.num_labels)
                )
        elif self.output_dim == 1:
            vals = score_fn(true_labels, (pred_labels>0.5)*1, average=None, labels=range(self.num_labels))
        else:
            if len(true_labels.shape) > 1 and true_labels.shape[-1] != 1:
                true_labels_ = np.argmax(true_labels, axis=1)
            else:
                true_labels_ = true_labels.squeeze()
            
            if len(pred_labels.shape) > 1 and pred_labels.shape[-1] != 1:
                pred_labels_ = np.argmax(pred_labels, axis=1)
            else:
                pred_labels_ = pred_labels.squeeze()
            
            vals = score_fn(
                true_labels_,
                pred_labels_,
                average=None,
                labels=range(self.num_labels)
            )
        if name not in score_dict:
            score_dict[name] = {}
        
        score_dict[name]['macro'] = sum(vals) / self.num_labels

        for i in range(self.num_labels):
            score_dict[name][i] = vals[i]
        
        return score_dict

    def eval_model(self, data=None, y_pred=None):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        '''

        if y_pred is not None:
            y_pred = np.reshape(a=y_pred, newshape=(-1,1))
            labels = self.get_labels(data=data)
            loss = 0
        else:
            y_pred, labels, loss = self.predict(data)

        score_dict = self.score(y_pred, labels)

        return score_dict, loss, y_pred

    def get_labels(self, data=None):
        if data is None:
            data = self.dataloader
        
        all_labels = torch.tensor([], device="cpu")
        for batch_n, batch_data in tqdm(enumerate(data)):
            label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            
            all_labels = torch.cat((all_labels, label_tensor))
        return all_labels.numpy()

    def predict(self, data=None):
        self.model.eval()
        self.text_input_model.eval()
        if not self.is_joint_text_topic and self.topic_input_model:
            self.topic_input_model.eval()
        
        partial_loss = 0
        all_y_pred = torch.tensor([], device="cpu")
        all_labels = torch.tensor([], device="cpu")
        
        if data is None:
            data = self.dataloader
        
        for batch_n, batch_data in tqdm(enumerate(data)):
            with torch.no_grad():
                #zero gradients before every optimizer step
                label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
                if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                    label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
                label_tensor = label_tensor.squeeze(dim=-1)

                if self.num_labels <= 2:
                    label_tensor = label_tensor.type(torch.FloatTensor)
                else:
                    label_tensor = label_tensor.type(torch.LongTensor)

                all_labels = torch.cat((all_labels, label_tensor))
                if self.use_cuda:
                    label_tensor = label_tensor.to("cuda")

                # get the embeddings for text and topic and creates a dict to pass as params to the model
                text_embeddings = self.text_input_model(**batch_data["text"])
                if self.is_joint_text_topic:
                    model_inputs = {
                        "input":text_embeddings,
                        "input_length": batch_data["text"]["input_length"],
                    }
                else:
                    model_inputs = {
                        "text_embeddings": text_embeddings,
                        "text_length": batch_data["text"]["input_length"],
                    }
                    if self.topic_input_model:
                        topic_embeddings = self.topic_input_model(**batch_data["topic"])
                        model_inputs.update({
                            "topic_embeddings": topic_embeddings,
                            "topic_length": batch_data["topic"]["input_length"]
                        })
                
                if self.is_ensemble:
                    clf_pred_label_1_tensor = torch.stack(batch_data["pred_label_1"]).T.type(torch.FloatTensor)
                    if len(clf_pred_label_1_tensor.shape) > 1 and clf_pred_label_1_tensor.shape[-1] != 1:
                        clf_pred_label_1_tensor = clf_pred_label_1_tensor.argmax(dim=1).reshape(-1,1)
                    clf_pred_label_1_tensor = clf_pred_label_1_tensor.squeeze(dim=-1)

                    clf_pred_label_2_tensor = torch.stack(batch_data["pred_label_2"]).T.type(torch.FloatTensor)
                    if len(clf_pred_label_2_tensor.shape) > 1 and clf_pred_label_2_tensor.shape[-1] != 1:
                        clf_pred_label_2_tensor = clf_pred_label_2_tensor.argmax(dim=1).reshape(-1,1)
                    clf_pred_label_2_tensor = clf_pred_label_2_tensor.squeeze(dim=-1)

                    model_inputs.update({
                        "clf1_pred": clf_pred_label_1_tensor,
                        "clf2_pred": clf_pred_label_2_tensor,
                    })
                
                y_pred = self.model(**model_inputs)
                if y_pred.squeeze(dim=-1).shape != torch.Size([]):
                    y_pred = y_pred.squeeze(dim=-1)

                # if self.use_cuda:
                all_y_pred = torch.cat((all_y_pred, y_pred.cpu()))
                
                graph_loss = self.loss_function(y_pred, label_tensor)
                if "sample_weight" in batch_data:
                    weight_lst = batch_data["sample_weight"]
                    if self.use_cuda:
                        weight_lst = weight_lst.to('cuda')
                    
                    graph_loss = torch.mean(graph_loss * weight_lst)
            
                partial_loss += graph_loss.item()

        avg_loss = partial_loss / max(batch_n, 1) #loss per batch
        return all_y_pred.numpy(), all_labels.numpy(), avg_loss#partial_loss
    
    def eval_and_print(self, data=None, data_name=None, y_pred=None):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged.
        Prints the results to the console.
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param data_name: the name of the data evaluating.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        # Passing data_name to eval_model as evaluation of adv model on train and dev are different
        scores, loss, y_pred = self.eval_model(data=data, y_pred=y_pred)
        print("Evaluating on \"{}\" data".format(data_name))
        for metric_name, metric_dict in scores.items():
            for class_name, metric_val in metric_dict.items():
                print(f"{metric_name}_{class_name}: {metric_val:.4f}", end="\t")
            print()

        return scores, loss, y_pred

    def score(self, pred_labels, true_labels):
        '''
        Helper Function to compute scores. Stores updated scores in
        the field "score_dict".
        :param pred_labels: the predicted labels
        :param true_labels: the correct labels
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        '''
        score_dict = dict()
        # calculate class-wise and macro-averaged F1
        score_dict = self.compute_scores(f1_score, true_labels, pred_labels, 'f', score_dict)
        # calculate class-wise and macro-average precision
        score_dict = self.compute_scores(precision_score, true_labels, pred_labels, 'p', score_dict)
        # calculate class-wise and macro-average recall
        score_dict = self.compute_scores(recall_score, true_labels, pred_labels, 'r', score_dict)

        return score_dict

class TOADTorchModelHandler(TorchModelHandler):
    def __init__(self, num_ckps=1, checkpoint_path='./data/checkpoints/', use_cuda=False, **params):

        TorchModelHandler.__init__(
            self,
            num_ckps=num_ckps,
            checkpoint_path=checkpoint_path,
            use_cuda=use_cuda,
            **params
        )

        self.adv_optimizer = params['adv_optimizer']
        self.tot_epochs = params['tot_epochs']
        self.initial_lr = params['initial_lr']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.num_constant_lr = params['num_constant_lr']

    def adjust_learning_rate(self, epoch):
        if epoch >= self.num_constant_lr:
            tot_epochs_for_calc = self.tot_epochs - self.num_constant_lr
            epoch_for_calc = epoch - self.num_constant_lr
            p = epoch_for_calc / tot_epochs_for_calc
            new_lr = self.initial_lr / ((1 + self.alpha * p) ** self.beta)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.adv_optimizer.param_groups:
                param_group['lr'] = new_lr

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        if self.epoch > 0:  # self.loss_function.use_adv:
            self.loss_function.update_param_using_p(self.epoch)  # update the adversarial parameter
        
        print("[{}] epoch {}".format(self.name, self.epoch))
        print("Adversarial parameter rho - {}".format(self.loss_function.adv_param))
        print("Learning rate - {}".format(self.get_learning_rate()))
        
        self.model.train()
        self.text_input_model.eval()
        if not self.is_joint_text_topic and self.topic_input_model:
            self.topic_input_model.eval()

        # clear the loss
        self.loss = 0.
        self.adv_loss = 0
        
        # TRAIN
        start_time = time.time()
        for batch_n, batch_data in tqdm(enumerate(self.dataloader)):
            # zero gradients before EVERY optimizer step3
            label_tensor = torch.stack(batch_data["label"]).T
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze()
            if len(label_tensor.shape) == 0:
                continue

            topic_tensor = torch.stack(batch_data["topic_label"]).T
            if len(topic_tensor.shape) > 1 and topic_tensor.shape[-1] != 1:
                topic_tensor = topic_tensor.argmax(dim=1).reshape(-1,1)
            topic_tensor = topic_tensor.squeeze()

            text_mask = batch_data["text"]["attention_mask"].type(torch.IntTensor)
            topic_mask = batch_data["topic"]["attention_mask"].type(torch.IntTensor)
            
            if self.use_cuda:
                label_tensor = label_tensor.to("cuda")
                topic_tensor = topic_tensor.to("cuda")

                text_mask = text_mask.to("cuda")
                topic_mask = topic_mask.to("cuda")

            self.model.zero_grad()

            # get the embeddings for text and topic and creates a dict to pass as params to the model
            text_embeddings = self.text_input_model(**batch_data["text"])
            topic_embeddings = self.topic_input_model(**batch_data["topic"])
            model_inputs = {
                "text_embeddings": text_embeddings,
                "text_length": batch_data["text"]["input_length"],
                "text_mask": text_mask,
                "topic_embeddings": topic_embeddings,
                "topic_length": batch_data["topic"]["input_length"],
                "topic_mask": topic_mask,
            }

            #apply the text and topic embeddings to the model
            pred_info = self.model(**model_inputs)

            pred_info['W'] = self.model.trans_layer.W
            pred_info['topic_i'] = topic_tensor         #Assigning topic indices to this dictionary element which is then used to calc adversarial loss on predicting train data topics

            # While training we want to compute adversarial loss.
            graph_loss_all, graph_loss_adv = self.loss_function(
                pred_info,
                label_tensor,
                compute_adv_loss=True
            )
            self.loss += graph_loss_all.item()
            self.adv_loss += graph_loss_adv.item()
            
            graph_loss_all.backward(retain_graph=True)  # NOT on adv. params
            self.optimizer.step()

            self.model.zero_grad()
            if True:  # self.loss_function.use_adv: - always do this, train adversary a bit first on it's own
                graph_loss_adv.backward()
                self.adv_optimizer.step()
                # only on adv params

        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1
        self.adjust_learning_rate(self.epoch)                # Adjusts the main and adversary optimizer learning rates using logic in base paper.

    def predict(self, data=None):
        self.model.eval()
        self.text_input_model.eval()
        if not self.is_joint_text_topic and self.topic_input_model:
            self.topic_input_model.eval()
        
        partial_main_loss = 0

        all_stance_pred = None
        all_stance_labels = None

        all_topic_pred = None
        all_topic_labels = None

        if data is None:
            data = self.dataloader
        
        for batch_n, batch_data in tqdm(enumerate(data)):
            with torch.no_grad():
                #zero gradients before every optimizer step
                label_tensor = torch.stack(batch_data["label"]).T
                if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                    label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
                label_tensor = label_tensor.squeeze()
                if len(label_tensor.shape) == 0:
                    continue

                topic_tensor = torch.stack(batch_data["topic_label"]).T
                if len(topic_tensor.shape) > 1 and topic_tensor.shape[-1] != 1:
                    topic_tensor = topic_tensor.argmax(dim=1).reshape(-1,1)
                topic_tensor = topic_tensor.squeeze()

                text_mask = batch_data["text"]["attention_mask"].type(torch.IntTensor)
                topic_mask = batch_data["topic"]["attention_mask"].type(torch.IntTensor)

                if batch_n:
                    all_stance_labels = torch.cat((all_stance_labels, label_tensor))
                    all_topic_labels = torch.cat((all_topic_labels, topic_tensor))
                else:
                    all_stance_labels = label_tensor
                    all_topic_labels = topic_tensor

                if self.use_cuda:
                    label_tensor = label_tensor.to("cuda")
                    topic_tensor = topic_tensor.to("cuda")

                    text_mask = text_mask.to("cuda")
                    topic_mask = topic_mask.to("cuda")

                # get the embeddings for text and topic and creates a dict to pass as params to the model
                text_embeddings = self.text_input_model(**batch_data["text"])
                topic_embeddings = self.topic_input_model(**batch_data["topic"])
                model_inputs = {
                    "text_embeddings": text_embeddings,
                    "text_length": batch_data["text"]["input_length"],
                    "text_mask": text_mask,
                    "topic_embeddings": topic_embeddings,
                    "topic_length": batch_data["topic"]["input_length"],
                    "topic_mask": topic_mask,
                }
                    
                #apply the text and topic embeddings to the model
                pred_info = self.model(**model_inputs)

                if batch_n:
                    all_stance_pred = torch.cat((all_stance_pred, pred_info["stance_pred"].cpu()))
                    all_topic_pred = torch.cat((all_topic_pred, pred_info["adv_pred"].cpu()))
                else:
                    all_stance_pred = pred_info["stance_pred"].cpu()
                    all_topic_pred = pred_info["adv_pred"].cpu()

                pred_info['W'] = self.model.trans_layer.W
                pred_info['topic_i'] = topic_tensor         #Assigning topic indices to this dictionary element which is then used to calc adversarial loss on predicting train data topics

                # While training we want to compute adversarial loss.
                graph_loss_all, graph_loss_adv = self.loss_function(
                    pred_info,
                    label_tensor,
                    compute_adv_loss=False
                )

                partial_main_loss += graph_loss_all.item()

        avg_loss = partial_main_loss / max(batch_n, 1) #loss per batch
        return all_stance_pred, all_stance_labels, all_topic_pred, all_topic_labels, avg_loss

    def eval_model(self, data=None, y_pred=None):
        # pred_topics and true_topics will be none while evaluating on dev set
        if y_pred is not None:
            stance_pred = np.reshape(a=y_pred, newshape=(-1,1))
            stance_labels = self.get_labels(data=data)
            loss = 0
        else:
            stance_pred, stance_labels, topic_pred, topic_labels, loss = self.predict(data)

        score_dict = self.score(stance_pred, stance_labels)

        return score_dict, loss, stance_pred

class AADTorchModelHandler(TorchModelHandler):
    def __init__(self, num_ckps=1, checkpoint_path='./data/checkpoints/', use_cuda=False, **params):

        TorchModelHandler.__init__(
            self,
            num_ckps=num_ckps,
            checkpoint_path=checkpoint_path,
            use_cuda=use_cuda,
            **params
        )

        self.tgt_dataloader = params["tgt_dataloader"]
        self.src_encoder = self.text_input_model
        self.tgt_encoder = params["tgt_text_input_model"]

        self.discriminator_loss_fn = params["discriminator_loss_fn"]
        self.discriminator_clip_value = params["discriminator_clip_value"]
        self.discriminator_optimizer = params["discriminator_optimizer"]
        
        self.tgt_encoder_optimizer = params["tgt_encoder_optimizer"]
        self.tgt_encoder_temperature = params["tgt_encoder_temperature"]
        self.tgt_encoder_loss_fn = params["tgt_encoder_loss_fn"]
        self.tgt_loss_alpha = params["tgt_loss_alpha"]
        self.tgt_loss_beta = params["tgt_loss_beta"]
        self.max_grad_norm = params["max_grad_norm"]

        self.KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')

    def pretrain_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print(f"[{self.name}] epoch {self.epoch}")
        self.model.train()
        self.src_encoder.train()
       
        self.loss = 0
        # partial_loss = 0
        start_time = time.time()

        for batch_n, batch_data in tqdm(enumerate(self.dataloader)):
            #zero gradients before every optimizer step
            label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze()
            
            if self.use_cuda:
                label_tensor = label_tensor.to("cuda")

            self.model.zero_grad()
            
            # get the embeddings for text and topic and creates a dict to pass as params to the model
            text_embeddings = self.src_encoder(**batch_data["text"])[:,-1,:]

            #apply the text and topic embeddings to the model
            y_pred = self.model(text_embeddings=text_embeddings)

            # calculate the loss, and backprogate it to update weights
            graph_loss = self.loss_function(y_pred.squeeze(), label_tensor)
            if "sample_weight" in batch_data:
                weight_lst = batch_data["sample_weight"]
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(graph_loss * weight_lst)
            
            graph_loss.backward()
            self.optimizer.step()

            #sum the loss
            # partial_loss += graph_loss.item()
            self.loss += graph_loss.item()

        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1

    def adapt_step(self):
        '''
        Runs one epoch of adapt process on this model.
        '''
        print(f"[{self.name}] epoch {self.epoch}")
        # set train state to the right models
        self.src_encoder.eval()
        self.model.classifier.eval()
        self.tgt_encoder.train()
        self.model.discriminator.train()
       
        self.discriminator_loss = 0
        # partial_loss = 0
        start_time = time.time()

        data_iter = tqdm(enumerate(zip(self.dataloader, self.tgt_dataloader)))
        for batch_n, (src_batch_data, tgt_batch_data) in data_iter:
            # make sure only equal size batches are used from both source and target domain
            if len(src_batch_data["label"][0]) != len(tgt_batch_data["label"][0]):
                continue

            # def get_label_tensor(x, use_cuda):
            #     x_tensor = torch.stack(x).T
            #     if len(x_tensor.shape) > 1 and x_tensor.shape[-1] != 1:
            #         x_tensor = x_tensor.argmax(dim=1).reshape(-1,1)
                
            #     if use_cuda:
            #         x_tensor = x_tensor.to("cuda")
            #     return x_tensor

            # stance_lbl_src = get_label_tensor(src_batch_data["label"], self.use_cuda)
            # stance_lbl_tgt = get_label_tensor(tgt_batch_data["label"], self.use_cuda)

            #zero gradients before every optimizer step
            self.model.zero_grad()
            
            # get the embeddings for src and tgt text using the tgt_encoder and concat them to pass to the discriminator
            with torch.no_grad():
                src_text_embeddings = self.src_encoder(**src_batch_data["text"])[:,-1,:]
            src_tgt_text_embeddings = self.tgt_encoder(**src_batch_data["text"])[:,-1,:]
            tgt_text_embeddings = self.tgt_encoder(**tgt_batch_data["text"])[:,-1,:]
            embeddings_concat = torch.cat((src_tgt_text_embeddings, tgt_text_embeddings), 0)

            # prepare real and fake label to calculate the discriminator loss
            domain_label_src = torch.ones(src_tgt_text_embeddings.size(0)).unsqueeze(1)
            domain_label_tgt = torch.zeros(tgt_text_embeddings.size(0)).unsqueeze(1)

            if self.use_cuda:
                domain_label_src = domain_label_src.to("cuda")
                domain_label_tgt = domain_label_tgt.to("cuda")
            
            domain_label_concat = torch.cat((domain_label_src, domain_label_tgt), 0)

            # predict on discriminator
            domain_pred_concat = self.model.discriminator(embeddings_concat.detach())

            # calculate the loss, and backprogate it to update weights
            graph_loss = self.discriminator_loss_fn(domain_pred_concat, domain_label_concat)
            if "sample_weight" in src_batch_data or "sample_weight" in tgt_batch_data:
                if "sample_weight" in src_batch_data:
                    weight_lst = src_batch_data["sample_weight"]
                else:
                    weight_lst = [1] * len(src_batch_data["label"])
                
                if "sample_weight" in tgt_batch_data:
                    weight_lst = tgt_batch_data["sample_weight"]
                else:
                    weight_lst = [1] * len(tgt_batch_data["label"])
            
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(graph_loss * weight_lst)
            
            graph_loss.backward()
            
            #clip the values if necessary
            if self.discriminator_clip_value:
                for p in self.model.discriminator.parameters():
                    p.data.clamp_(-self.discriminator_clip_value, self.discriminator_clip_value)

            self.discriminator_optimizer.step()
            #sum the loss
            self.discriminator_loss += graph_loss.item()

            # zero gradients for optimizer
            self.tgt_encoder_optimizer.zero_grad()
            T = self.tgt_encoder_temperature

            # predict on discriminator
            pred_tgt = self.model.discriminator(tgt_text_embeddings)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = torch.nn.functional.softmax(self.model.classifier(src_text_embeddings) / T, dim=-1)
            tgt_prob = torch.nn.functional.log_softmax(self.model.classifier(src_tgt_text_embeddings) / T, dim=-1)

            kd_loss = self.KLDivLoss(tgt_prob, src_prob.detach()) * T * T
            tgt_encoder_loss = self.tgt_encoder_loss_fn(pred_tgt, domain_label_src)
            loss_tgt = self.tgt_loss_alpha * tgt_encoder_loss + self.tgt_loss_beta * kd_loss

            # multiply by the weights, if any, and calculate the mean value
            if "sample_weight" in tgt_batch_data:
                tgt_weight_lst = tgt_batch_data["sample_weight"]
                if self.use_cuda:
                    tgt_weight_lst = tgt_weight_lst.to('cuda')
                loss_tgt = torch.mean(loss_tgt * tgt_weight_lst)

            # compute loss for target encoder
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(self.tgt_encoder.parameters(), self.max_grad_norm)
            # optimize target encoder
            self.tgt_encoder_optimizer.step()

        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1

    def predict(self, data=None, encoder=None):
        self.model.eval()
        self.src_encoder.eval()
        self.tgt_encoder.eval()
        
        partial_main_loss = 0

        all_stance_pred = None
        all_stance_labels = None

        all_topic_pred = None
        all_topic_labels = None

        if data is None:
            data = self.dataloader
        
        if encoder is None:
            encoder = self.tgt_encoder

        for batch_n, batch_data in tqdm(enumerate(data)):
            with torch.no_grad():
                #zero gradients before every optimizer step
                label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
                if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                    label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
                label_tensor = label_tensor.squeeze()

                if batch_n:
                    all_stance_labels = torch.cat((all_stance_labels, label_tensor))
                else:
                    all_stance_labels = label_tensor

                if self.use_cuda:
                    label_tensor = label_tensor.to("cuda")
            
                # get the embeddings for text and topic and creates a dict to pass as params to the model
                text_embeddings = encoder(**batch_data["text"])[:,-1,:]

                #apply the text and topic embeddings to the model
                y_pred = self.model(text_embeddings=text_embeddings)

                if batch_n:
                    all_stance_pred = torch.cat((all_stance_pred, y_pred.cpu()))
                else:
                    all_stance_pred = y_pred.cpu()

                # calculate the loss, and backprogate it to update weights
                graph_loss = self.loss_function(y_pred.squeeze(), label_tensor)
                if "sample_weight" in batch_data:
                    weight_lst = batch_data["sample_weight"]
                    if self.use_cuda:
                        weight_lst = weight_lst.to('cuda')
                    graph_loss = torch.mean(graph_loss * weight_lst)
                
                partial_main_loss += graph_loss.item()

            #sum the loss
            # partial_loss += graph_loss.item()
        avg_loss = partial_main_loss / max(batch_n, 1) #loss per batch

        return all_stance_pred, all_stance_labels, avg_loss


import faiss
gpu_id = 0

class JointCLTorchModelHandler(TorchModelHandler):
    def __init__(self, num_ckps=1, checkpoint_path='./data/checkpoints/', use_cuda=False, **params):

        TorchModelHandler.__init__(
            self,
            num_ckps=num_ckps,
            checkpoint_path=checkpoint_path,
            use_cuda=use_cuda,
            **params
        )
        
        self.bert_dim = params["bert_dim"]
        self.cluster_result = None
        self.cluster_times = params["cluster_times"]
        self.device = params["device"]
        self.logits_loss_fn = params["logits_loss_fn"]
        self.prototype_loss_weight = params["prototype_loss_weight"]
        self.stance_loss_fn = params["stance_loss_fn"]
        self.stance_loss_weight = params["stance_loss_weight"]
        self.target_loss_fn = params["target_loss_fn"]
        self.temperature = params["temperature"]
        self.train_loader_prototype = params["train_loader_prototype"]
        self.output_dim = self.num_labels
        
    def compute_features(self, train_loader):
        print('Computing features...')
        self.model.eval()
        features = torch.zeros(len(train_loader.dataset),self.bert_dim)
        if self.use_cuda:
            features = features.cuda()
        
        for batch in tqdm(train_loader):
            
            index = batch['index']
            input_features = [
                batch["text"]["input_ids"],
                batch["text"]["is_topic_mask"],
            ]

            if self.use_cuda:
                for k, inp_feat in enumerate(input_features):
                    input_features[k] = inp_feat.to("cuda")
            
            with torch.no_grad():
                feature = self.model.prototype_encode(input_features)
                feature = feature.squeeze(dim=1)
                features[index] = feature

        return features.cpu()

    def run_kmeans(self, x):
        print('performing kmeans clustering')
        results = {
            'im2cluster': [],
            'centroids': [],
            'density': []
        }

        for seed, num_cluster in enumerate(self.num_cluster):
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 1000
            clus.min_points_per_centroid = 10

            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = gpu_id
            index = faiss.GpuIndexFlatL2(res, d, cfg)

            clus.train(x, index)

            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im,i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i,dist in enumerate(Dcluster):
                if len(dist)>1:
                    d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                    density[i] = d
                    
                    #if cluster only has one point, use the max to estimate its concentration

            dmax = density.max()
            for i,dist in enumerate(Dcluster):
                if len(dist)<=1:
                    density[i] = dmax

            density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
            density = self.temperature*density/density.mean()  #scale the mean to temperature

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

            im2cluster = torch.LongTensor(im2cluster).cuda()
            density = torch.Tensor(density).cuda()

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)

        return results

    def run_prototype(self,train_loader):
        self.warmup_epoch = 0
        self.num_cluster = [25]

        features = self.compute_features(train_loader)

        cluster_result = {
            'im2cluster': [],
            'centroids': [],
            'density': [],
        }

        for num_cluster in self.num_cluster:
            if self.use_cuda:
                cluster_result['im2cluster'].append(torch.zeros(len(train_loader.dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),self.bert_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
            else:
                cluster_result['im2cluster'].append(torch.zeros(len(train_loader.dataset),dtype=torch.long))
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),self.bert_dim))
                cluster_result['density'].append(torch.zeros(int(num_cluster)))
        features = features.numpy()
        cluster_result = self.run_kmeans(features)

        return cluster_result

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print(f"[{self.name}] epoch {self.epoch}")
        self.model.train()
        
        self.loss = 0
        start_time = time.time()

        for batch_n, batch_data in tqdm(enumerate(self.dataloader)):

            if batch_n % int(len(self.dataloader)/self.cluster_times) == 0:
                cluster_result = self.run_prototype(self.train_loader_prototype)

            label_tensor = torch.stack(batch_data["label"]).T.type(torch.LongTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze()
            
            topic_tensor = torch.stack(batch_data["topic_label"]).T.type(torch.FloatTensor)
            if len(topic_tensor.shape) > 1 and topic_tensor.shape[-1] != 1:
                topic_tensor = topic_tensor.argmax(dim=1).reshape(-1,1)
            topic_tensor = topic_tensor.squeeze()
            
            stance_topic_tensor = label_tensor + (self.model.num_labels * topic_tensor)

            # index = batch_data["index"]
            input_features = [
                batch_data["text"]["input_ids"],
                batch_data["text"]["is_topic_mask"],
            ]

            self.model.zero_grad()

            if self.use_cuda:
                label_tensor = label_tensor.to("cuda")
                topic_tensor = topic_tensor.to("cuda")

                for k, inp_feat in enumerate(input_features):
                    input_features[k] = inp_feat.to("cuda")
            
            feature = self.model.prototype_encode(input_features)
            
            logits, node_for_con = self.model(input_features+[cluster_result['centroids']])
            self.cluster_result = [cluster_result['centroids']]

            if cluster_result is not None:

                for n, im2cluster in enumerate(cluster_result['im2cluster']):
                    # pos_proto_id = im2cluster[index]

                    # prototype_loss = self.target_loss_fn(node_for_con, label_tensor, topic_tensor, pos_proto_id)
                    prototype_loss = self.target_loss_fn(node_for_con, stance_topic_tensor)
                    stance_loss = self.stance_loss_fn(feature, label_tensor)
            else:
                prototype_loss = 0.0

            logits_loss = self.logits_loss_fn(logits.squeeze(), label_tensor)
            
            # print("*"*20)
            # print("logits_loss", logits_loss)
            # print("stance_loss", stance_loss, stance_loss * self.stance_loss_weight)
            # print("prototype_loss", prototype_loss, prototype_loss * self.prototype_loss_weight)
            graph_loss = logits_loss + stance_loss * self.stance_loss_weight + prototype_loss * self.prototype_loss_weight
            # print(f"Train step: graph_loss: {graph_loss:.5}  logit_loss:{logits_loss:.5}, stance loss: {stance_loss:.5}, prototype loss: {prototype_loss:.5} ")
            # print("logits",logits)
            graph_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.loss += graph_loss.item()
        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1
    
    def predict(self, data=None):
        self.model.eval()
        
        partial_loss = 0
        all_y_pred = torch.tensor([], device="cpu")
        all_labels = torch.tensor([], device="cpu")
        
        if data is None:
            data = self.dataloader
        
        for batch_n, batch_data in tqdm(enumerate(data)):
            with torch.no_grad():
                label_tensor = torch.stack(batch_data["label"]).T.type(torch.LongTensor)
                if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                    label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
                label_tensor = label_tensor.squeeze(-1)
                
                input_features = [
                    batch_data["text"]["input_ids"],
                    batch_data["text"]["is_topic_mask"],
                ]

                all_labels = torch.cat((all_labels, label_tensor))
                if self.use_cuda:
                    label_tensor = label_tensor.to("cuda")
                    
                    for k, inp_feat in enumerate(input_features):
                        input_features[k] = inp_feat.to("cuda")

                y_pred, _ = self.model(input_features+self.cluster_result)
                graph_loss = self.logits_loss_fn(y_pred.squeeze(-1), label_tensor)

                # if self.use_cuda:
                all_y_pred = torch.cat((all_y_pred, y_pred.cpu()))
                
                partial_loss += graph_loss.item()

        avg_loss = partial_loss / max(batch_n, 1) #loss per batch
        print("0.0 - 0.1", (all_y_pred.numpy() <= 0.1).sum(axis=0))
        print("0.1 - 0.2", ((all_y_pred.numpy() > 0.1) * (all_y_pred.numpy()<= 0.2)).sum(axis=0))
        print("0.2 - 0.3", ((all_y_pred.numpy() > 0.2) * (all_y_pred.numpy()<= 0.3)).sum(axis=0))
        print("0.3 - 0.4", ((all_y_pred.numpy() > 0.3) * (all_y_pred.numpy()<= 0.4)).sum(axis=0))
        print("0.4 - 0.5", ((all_y_pred.numpy() > 0.4) * (all_y_pred.numpy()<= 0.5)).sum(axis=0))
        print("0.5 - 0.6", ((all_y_pred.numpy() > 0.5) * (all_y_pred.numpy()<= 0.6)).sum(axis=0))
        print("0.6 - 0.7", ((all_y_pred.numpy() > 0.6) * (all_y_pred.numpy()<= 0.7)).sum(axis=0))
        print("0.7 - 0.8", ((all_y_pred.numpy() > 0.7) * (all_y_pred.numpy()<= 0.8)).sum(axis=0))
        print("0.8 - 0.9", ((all_y_pred.numpy() > 0.8) * (all_y_pred.numpy()<= 0.9)).sum(axis=0))
        print("0.9 - 1.0", (all_y_pred.numpy() > 0.9).sum(axis=0))
        return all_y_pred.numpy(), all_labels.numpy(), avg_loss#partial_loss

import re
class LLMTorchModelHandler(TorchModelHandler):
    def __init__(self, num_ckps=1, checkpoint_path='./data/checkpoints/', use_cuda=False, **params):

        TorchModelHandler.__init__(
            self,
            num_ckps=num_ckps,
            checkpoint_path=checkpoint_path,
            use_cuda=use_cuda,
            **params
        )
        
        self.is_llm = True
        self.is_hfllm = params["is_hfllm"]
        self.model = params["model"]
        self.model_params = params.get("model_params", {})
        self.dataset = params["dataset"]
        self.tokenizer = self.dataset.tokenizer
        self.tokenized_input = self.tokenizer is None
        self.output_format = params["output_format"]
        self.output_max_score = int(params.get("output_max_score", "10"))
        self.output_parser = params.get("output_parser", self._output_parser_fn)
        self.save_every_n_batches = int(params.get("save_every_n_batches", "0"))
        self.output_err_default = float(params.get("output_err_default", "0.0"))
        
        if self.output_format == "score":
            self.output_err_default = float(self.output_err_default)
        
        self.output_class_order = params.get("output_class_order")
        self.output_class_map = None
        if self.output_class_order is not None:
            self.output_class_order = self.output_class_order.split(",")
            self.output_class_map = [self.dataset.tgt2vec[k] for k in self.output_class_order]

        if len(self.model_params) == 0:
            self.model_params["max_new_tokens"] = 32
    def _output_parser_fn(self, output):
        output = output.strip().lower()

        try:
            if self.output_format == "set":
                possible_values = '|'.join(self.dataset.tgt2vec.keys())
                set_regex = re.search(f"({possible_values})", output)

                prediction = output[set_regex.start():set_regex.end()].lower()
                prediction = self.dataset.convert_lbl_to_vec(prediction, self.dataset.tgt2vec)
                if len(prediction) > 1:
                    prediction = (np.argmax(prediction)+0.5)/self.num_labels
                else:
                    prediction = prediction[0]/self.num_labels
            
            elif self.output_format == "score":
                score_regex = re.search("([0-9]+)(/[0-9]+)*", output)
                
                prediction = output[score_regex.start():score_regex.end()]
                if "/" in prediction:
                    prediction = prediction.split("/")
                    prediction = float(prediction[0])/float(prediction[1])
                else:
                    prediction = float(prediction)/self.output_max_score
                
                if self.output_class_map is not None:
                    # convert float back to integer (up to num_labels)
                    prediction = math.floor(prediction*self.num_labels)
                    # convert int to vec (consistent with dataset)
                    prediction = self.output_class_map[prediction]
                    # convert vec to new int (order in dataset)
                    prediction = (np.argmax(prediction)+0.5)/self.num_labels

        except Exception as ex:
            prediction = None

        return prediction
    
    def predict(self, data=None):
        all_y_pred = []
        all_labels = []
        all_indices = []
        errors_ = []
        if data is None:
            data = self.dataloader
        
        call_count = 0
        for batch_n, batch_data in tqdm(enumerate(data)):
            if self.is_hfllm: # pytorch=2.2+
                label_tensor = torch.Tensor(batch_data["label"])
            else: #pytorch==1.17
                label_tensor = torch.stack(batch_data["label"]).T

            label_tensor = label_tensor.type(torch.LongTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze(-1).numpy()
                
            all_labels += label_tensor.tolist()
            all_indices += batch_data["index"]

            if self.model.model_type in ["llama_cpp", "hf_api"]:
                prompt_list = batch_data["prompt"]

                str_output_list = []
                for prompt_ in prompt_list:
                    str_output_list += [
                        self.model(
                            prompt = prompt_,
                            params = self.model_params,
                        ).replace(prompt_, "")
                    ]
                
                # if self.model.model_type == "hf_api" and call_count >= 180:
                    # call_count += 1
                    # print(f"180 calls made, Waiting 11 minutes... {datetime.datetime.now()}")
                    # for i in tqdm(range(11*60), leave=False, desc="Waiting 11 minutes"):
                    #     time.sleep(1)
                    
            elif self.model.model_type == "hugging_face":
                prompt_ids_list = {
                    "input_ids": torch.stack(batch_data["input_ids"]).reshape(-1, self.dataset.max_prompt_length).to("cuda"),
                    "attention_mask": torch.stack(batch_data["attention_mask"]).reshape(-1, self.dataset.max_prompt_length).to("cuda"),
                }
                
                # str_output_list = []
                # for prompt_ids in prompt_ids_list:
                output_tensor = self.model(
                    **prompt_ids_list,
                    **self.model_params,
                    # params = self.model_params,
                )
                
                output_wo_echo = []
                for i in range(len(output_tensor)):
                    output_wo_echo += [output_tensor[i][prompt_ids_list["input_ids"][i].shape[-1]:]]

                str_output_list = self.dataset.decode_tokens(output_wo_echo)
                # if isinstance(decoded_out, list):
                #     str_output_list += decoded_out
                # else:
                #     str_output_list += [decoded_out]

            for idx, str_output, prompt_ in zip(batch_data["index"], str_output_list, batch_data["prompt"]):
                parsed_out = self.output_parser(str_output)
                if parsed_out is None:
                    parsed_out = self.output_err_default
                    errors_ += [(float(idx), prompt_, str_output)]

                all_y_pred += [parsed_out]

            if self.save_every_n_batches and batch_n % self.save_every_n_batches == 0:
                out_dict = {
                    "pred": np.array(all_y_pred).tolist(),
                    "index": np.array(all_indices).tolist(),
                }
                os.makedirs(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/", exist_ok=True)
                with open(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/{self.name}.ckp", mode="w", encoding="utf-8") as f_:
                    json.dump(out_dict, f_)

        # save the last batch that is not full size
        out_dict = {
            "pred": np.array(all_y_pred).tolist(),
            "index": np.array(all_indices).tolist(),
        }
        os.makedirs(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/", exist_ok=True)
        with open(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/{self.name}.ckp", mode="w", encoding="utf-8") as f_:
            json.dump(out_dict, f_)

        if len(errors_) > 0:
            os.makedirs(f"{self.checkpoint_path}/llama_cpp_errors/", exist_ok=True)
            with open(f"{self.checkpoint_path}/llama_cpp_errors/{self.name}.err", mode="w", encoding="utf-8") as f_:
                json.dump(errors_, f_)

        return np.array(all_y_pred), np.array(all_labels), None
    