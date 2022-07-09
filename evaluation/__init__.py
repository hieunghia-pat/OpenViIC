from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

def compute_language_scores(captions_gt, captions_gen):
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    for caption_gt, caption_gen in zip(captions_gt, captions_gen):
        if len(caption_gt) < len(caption_gen):
            caption_gen = caption_gen[:len(caption_gt)]
        
        if len(caption_gt) > len(caption_gen):
            delta_len = len(caption_gt) - len(caption_gen)
            caption_gen = caption_gen + ["<pad>"]*delta_len

        accuracy += accuracy_score(caption_gt, caption_gen)
        precision += precision_score(caption_gt, caption_gen, average="micro")
        recall += recall_score(caption_gt, caption_gen, average="micro")
        f1 += f1_score(caption_gt, caption_gen, average="micro")

    return {
        "accuracy": accuracy / len(captions_gt),
        "precision": precision / len(captions_gt),
        "recall": recall / len(captions_gt),
        "f1": f1 / len(captions_gt)
    }