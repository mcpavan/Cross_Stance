import torch, time
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
        self.is_joint_text_topic = "topic_input_model" not in params

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
        if not self.is_joint_text_topic:
            self.topic_input_model.eval()
        
        self.loss = 0
        # partial_loss = 0
        start_time = time.time()

        for batch_n, batch_data in tqdm(enumerate(self.dataloader)):
            #zero gradients before every optimizer step
            label_tensor = batch_data["label"][0].reshape(-1,1).type(torch.FloatTensor)
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
                topic_embeddings = self.topic_input_model(**batch_data["topic"])
                model_inputs = {
                    "text_embeddings": text_embeddings,
                    "text_length": batch_data["text"]["input_length"],
                    "topic_embeddings": topic_embeddings,
                    "topic_length": batch_data["topic"]["input_length"]
                }

            #apply the text and topic embeddings to the model
            y_pred = self.model(**model_inputs)

            # calculate the loss, and backprogate it to update weights
            graph_loss = self.loss_function(y_pred, label_tensor)
            if "sample_weight" in batch_data:
                if self.use_cuda:
                    weight_lst = batch_data["sample_weight"].to('cuda')
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
        if not self.is_joint_text_topic:
            self.topic_input_model.eval()
        
        partial_loss = 0
        all_y_pred = torch.tensor([], device="cpu")
        all_labels = torch.tensor([], device="cpu")
        
        if data is None:
            data = self.dataloader
        
        for batch_n, batch_data in tqdm(enumerate(data)):
            with torch.no_grad():
                #zero gradients before every optimizer step
                label_tensor = batch_data["label"][0].reshape(-1,1).type(torch.FloatTensor)
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
                    topic_embeddings = self.topic_input_model(**batch_data["topic"])
                    model_inputs = {
                        "text_embeddings": text_embeddings,
                        "text_length": batch_data["text"]["input_length"],
                        "topic_embeddings": topic_embeddings,
                        "topic_length": batch_data["topic"]["input_length"]
                    }
                
                y_pred = self.model(**model_inputs)
                if self.use_cuda:
                    all_y_pred = torch.cat((all_y_pred, y_pred.cpu()))
                
                graph_loss = self.loss_function(y_pred, label_tensor)
                if "sample_weight" in batch_data:
                    if self.use_cuda:
                        weight_lst = batch_data["sample_weight"].to('cuda')
                    graph_loss = torch.mean(graph_loss * weight_lst)
            
                partial_loss += graph_loss.item()

        # avg_loss = partial_loss / batch_n #loss per batch
        return all_y_pred.numpy(), all_labels.numpy(), partial_loss
    
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
                print(f"{metric_name}_{class_name}: {metric_val:.2f}", end="\t")
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