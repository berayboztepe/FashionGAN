from torchvision.models.inception import inception_v3
from scipy.stats import entropy
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=10):
    N = len(imgs)
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i in range(0, N, batch_size):
        batch = imgs[i:i + batch_size].type(dtype)
        batchv = torch.autograd.Variable(batch)
        preds[i:i + batch_size] = get_pred(batchv)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        split_scores.append(np.exp(kl))

    return np.mean(split_scores), np.std(split_scores)

def fid_score(real_images, generated_images, batch_size=50, cuda=True):
    inception_model = inception_v3(pretrained=True, transform_input=False)
    if cuda:
        inception_model.cuda()
    inception_model.eval()
    
    def get_activations(images):
        pred_arr = np.zeros((len(images), 2048))
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            if cuda:
                batch = batch.cuda()
            pred = inception_model(batch)[0]
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_arr[i:i + batch_size] = pred.cpu().data.numpy().reshape(batch.size(0), -1)
        return pred_arr

    act1 = get_activations(real_images)
    act2 = get_activations(generated_images)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
