import torch, time, numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import os

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
            # label_tensor = batch_data["label"][0].reshape(-1,1).type(torch.FloatTensor)
            label_tensor = torch.stack(batch_data["label"]).T
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            
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
                
            #apply the text and topic embeddings to the model
            y_pred = self.model(**model_inputs)

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
        vals = score_fn(true_labels, (pred_labels>0.5)*1, average=None, labels=range(self.num_labels))
        if name not in score_dict:
            score_dict[name] = {}
        
        score_dict[name]['macro'] = sum(vals) / self.num_labels

        for i in range(self.num_labels):
            score_dict[name][i] = vals[i]
        
        return score_dict

    def eval_model(self, data=None):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        '''
        y_pred, labels, loss = self.predict(data)
        score_dict = self.score(y_pred, labels)

        return score_dict, loss

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
                label_tensor = torch.stack(batch_data["label"]).T
                if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                    label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
                
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
                    
                y_pred = self.model(**model_inputs)
                # if self.use_cuda:
                all_y_pred = torch.cat((all_y_pred, y_pred.cpu()))
                
                graph_loss = self.loss_function(y_pred, label_tensor)
                if "sample_weight" in batch_data:
                    weight_lst = batch_data["sample_weight"]
                    if self.use_cuda:
                        weight_lst = weight_lst.to('cuda')
                    
                    graph_loss = torch.mean(graph_loss * weight_lst)
            
                partial_loss += graph_loss.item()

        avg_loss = partial_loss / batch_n #loss per batch
        return all_y_pred.numpy(), all_labels.numpy(), avg_loss#partial_loss
    
    def eval_and_print(self, data=None, data_name=None):
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
        scores, loss = self.eval_model(data=data)
        print("Evaluating on \"{}\" data".format(data_name))
        for metric_name, metric_dict in scores.items():
            for class_name, metric_val in metric_dict.items():
                print(f"{metric_name}_{class_name}: {metric_val}", end="\t")
            print()

        return scores, loss

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

            topic_tensor = torch.stack(batch_data["topic_label"]).T
            if len(topic_tensor.shape) > 1 and topic_tensor.shape[-1] != 1:
                topic_tensor = topic_tensor.argmax(dim=1).reshape(-1,1)

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

                topic_tensor = torch.stack(batch_data["topic_label"]).T
                if len(topic_tensor.shape) > 1 and topic_tensor.shape[-1] != 1:
                    topic_tensor = topic_tensor.argmax(dim=1).reshape(-1,1)
                
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

        avg_loss = partial_main_loss / batch_n #loss per batch
        return all_stance_pred, all_stance_labels, all_topic_pred, all_topic_labels, avg_loss

    def eval_model(self, data=None):
        # pred_topics and true_topics will be none while evaluating on dev set
        stance_pred, stance_labels, topic_pred, topic_labels, loss = self.predict(data)
        score_dict = self.score(stance_pred, stance_labels)

        return score_dict, loss
