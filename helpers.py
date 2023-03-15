import numpy as np
import matplotlib.pyplot as plt
import scipy
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
import json
from tqdm.auto import tqdm

QA_MAX_ANSWER_LENGTH = 30

def augment_train_set(examples, augmenter):
    # print("here")
    all_premises = [] + examples['premise']
    all_hyp = [] + examples['hypothesis']
    all_labels = [] + examples['label']
    # print(len(examples['premise']))
    # print("Ex:" + str(examples['premise']))
    # return
    for i, example in enumerate(examples['premise']):
        # print(i, examples['premise'])
        premises = augmenter.augment(example)
        all_premises += premises
        all_hyp += [examples['hypothesis'][i] for j in range(len(premises))]
        all_labels += [examples['label'][i] for j in range(len(premises))]
    # print("part 1")
    for i, example in enumerate(examples['hypothesis']):
        hypothesis = augmenter.augment(example)
        all_premises += [examples['premise'][i] for j in range(len(hypothesis))]
        all_hyp += hypothesis
        all_labels += [examples['label'][i] for j in range(len(hypothesis))]

        # all_premises += [example['premise']] + premises + [example['premise'] for i in range(len(hypothesis))]
        # all_hyp += [example[hypothesis]] + hypothesis + [example['hypothesis'] for i in range(len(premises))]
        # all_labels += [example[label] for i in range(1+len(premises)+len(hypothesis))]

    # print("done")
    return {'premise' : all_premises, 'hypothesis': all_hyp, 'label': all_labels}

    # [example] + [{'premise': premises[i], 'hypothesis': example['hypothesis'], 'label': example['label']} for i in range(len(premises))] \
    # + [{'premise': example['premise'], 'hypothesis': hypothesis[i], 'label': example['label']} for i in range(len(hypothesis))]

def stat_test(train_dataset, tokenizer, outFile, outFile2):
    i = 0
    word_count_per_class = dict()
    for sample in train_dataset:
        process_dataset_nli(word_count_per_class, sample, tokenizer)
        i+=1
        if (i%1000==0):
            print(i)
    # train_dataset.map(lambda exs: process_dataset_nli(word_count_per_class, exs, tokenizer), batched=True, num_proc=NUM_PREPROCESSING_WORKERS)
    print(len(word_count_per_class))
    biased_words = []
    biased_words_labels = dict()
    alpha = 0.01/len(word_count_per_class)
    print(alpha)
    z_scores = dict()
    max_n = 0
    for word in word_count_per_class:
        counts = word_count_per_class[word]
        z_scores[word] = dict()
        n = counts[0] + counts[1] + counts[2]
        if (n>max_n):
            max_n = n
        if (n<20):
            continue
        for label in counts:
            p_hat = counts[label]/n
            p0 = 1/3
            zScore = (p_hat - p0)/(np.sqrt(p0*(1-p0)/n))
            z_scores[word][label] = zScore
            p_value = scipy.stats.norm.sf(abs(zScore))
            if (p_value < alpha):
                biased_words.append(word)
    for word in biased_words:
        highLabel = 0
        if (z_scores[word][1] > z_scores[word][0]):
            highLabel = 1
        if (z_scores[word][2] > z_scores[word][highLabel]):
            highLabel = 2
        biased_words_labels[word] = highLabel
    # write_stat_test_dataset(biased_words_labels, z_scores, outFile, outFile2)
    plot(word_count_per_class, z_scores, alpha)
    # plot()
    print("biased: " + str(len(biased_words)))
    print("max n: " + str(max_n))
    return [word_count_per_class, z_scores]

def plot(word_count_per_class, z_scores, alpha):
# def plot():
    x = []
    y = [[],[],[]]
    for word in word_count_per_class:
        n = word_count_per_class[word][0] + word_count_per_class[word][1] + word_count_per_class[word][2]
        if (n<20):
            continue
        x.append(n)
        for label in word_count_per_class[word]:
            p_hat = word_count_per_class[word][label]/n
            if (p_hat > 0.3):
                p_hat = p_hat - np.random.rand()*0.2
            y[label].append(p_hat)

    z = 4.9909
    n = np.linspace(20, 1439104, 143910)
    p = z/(2*np.sqrt(n)) + 0.35
     
    fig = plt.figure(figsize = (10, 10))
    # Create the plot
    plt.scatter(x,y[0], c="blue", label='entailment', s=10)
    plt.scatter(x,y[1], c="red", label='neutral',s=10)
    plt.scatter(x,y[2], c="green", label='contradiction',s=10)
    # hold on
    # plt.plot(x,y)
    plt.ylim(0,1)
    plt.semilogx(n, p)
    plt.legend()
    # hold off
     
    # Show the plot
    plt.show()
def write_stat_test_dataset(biased_words, z_scores, outFile, outFile2):
    class_scores = [dict(), dict(), dict()]
    for word in biased_words:
        class_scores[biased_words[word]][word] = z_scores[word][biased_words[word]]
    class_samples = []
    # premise = []
    # hypothesis = []
    # class_labels = []
    # class_labels2 = []

    fileOutput1 = []
    fileOutput2 = []
    for label in range(3):
        # print(class_scores[label].items())
        sorted_words = sorted(class_scores[label].items(), key=lambda kv: kv[1])
        if (len(sorted_words) < 100):
            print("Less words for class" + str(label))
        top_50 = sorted_words[:50]
        bottom_50 = sorted_words[-50:]
        class_samples.append([top_50, bottom_50])
        for word in top_50: 
            d1 = {'premise': word[0], 'hypothesis': "", 'label': label}
            d2 = {'premise': "", 'hypothesis': word[0], 'label': label}
            fileOutput1.append(d1)
            fileOutput1.append(d2)
            d1 = {'premise': word[0], 'hypothesis': "", 'label': label+1}
            d2 = {'premise': "", 'hypothesis': word[0], 'label': label+1}
            fileOutput2.append(d1)
            fileOutput2.append(d2)
            # premise.append(word[0])
            # hypothesis.append("")
            # premise.append("")
            # hypothesis.append(word[0])
            # class_labels.append(label)
            # class_labels.append(label)
            # class_labels2.append(label+1)
            # class_labels2.append(label+1)
        for word in bottom_50:
            d1 = {'premise': word[0], 'hypothesis': "", 'label': label}
            d2 = {'premise': "", 'hypothesis': word[0], 'label': label}
            fileOutput1.append(d1)
            fileOutput1.append(d2)
            d1 = {'premise': word[0], 'hypothesis': "", 'label': -1*(label+1)}
            d2 = {'premise': "", 'hypothesis': word[0], 'label': -1*(label+1)}
            fileOutput2.append(d1)
            fileOutput2.append(d2)

            # premise.append(word[0])
            # hypothesis.append("")
            # premise.append("")
            # hypothesis.append(word[0])
            # class_labels.append(label)
            # class_labels.append(label)
            # class_labels2.append(-1*(label+1))
            # class_labels2.append(-1*(label+1))

    with open(outFile, 'w') as file:
        for line in fileOutput1:
            json_string = json.dumps(line)
            print(json_string, file=file)
    with open(outFile2, 'w') as file:
        for line in fileOutput2:
            json_string = json.dumps(line)
            print(json_string, file=file)
    
def abs_val_label(example):
    example["label"] = abs(example["label"])-1
    return example
def process_dataset_nli(word_count_per_class_dict, examples, tokenizer):
    # max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    normalizer = tokenizer.backend_tokenizer.normalizer
    pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    # print(examples)    
    for strType in ['premise', 'hypothesis']:
        curSet = examples[strType]
        temp = curSet
        # for index, temp in enumerate(curSet):
        temp = normalizer.normalize_str(temp)
        temp = pre_tokenizer.pre_tokenize_str(temp)
        # print(temp)
        temp = map(lambda x: x[0], temp) # Extract words only - we don't care about the offsets
        # print(temp)
        for word in temp:
            curVal = word_count_per_class_dict.get(word, {0:0, 1:0, 2:0})
            curVal[examples['label']] += 1;
            # print(curVal)
            word_count_per_class_dict[word] = curVal
    # print(word_count_per_class_dict)
    return
# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    # print(examples)
    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                            -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                          -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                                     end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds = postprocess_qa_predictions(eval_examples,
                                                    eval_dataset,
                                                    output.predictions)
            formatted_predictions = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples]

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions,
                               label_ids=references)
            )

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state,
                                                         self.control, metrics)
        return metrics