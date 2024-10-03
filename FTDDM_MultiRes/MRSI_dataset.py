from torch.utils.data import DataLoader, Dataset
import pathlib
import numpy as np
import logging
import torch
from .utils import utils
from torch.utils.data.sampler import Sampler
import random
import math
import blobfile as bf
from mpi4py import MPI


def load_data(
    *, patients, batch_size, class_cond=False):
    """
    For a dataset, create a generator over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    '''
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    '''
    dataset = MRSI_Dataset(patients, class_cond)
    met_sampler = Met_Sampler(batch_size=batch_size, length=dataset.__len__())
    loader = DataLoader(dataset, batch_sampler=met_sampler, num_workers=0)
    while True:
        yield from loader


def load_testdata(patients, batch_size, class_cond):
    dataset = MRSI_Dataset(patients, class_cond)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return testloader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class MRSI_Dataset(Dataset):
    def __init__(self, patients, class_cond):
        self.patients = patients
        self.class_cond = class_cond

        self.patient_list = ['Patient' + str(self.patients[i]) for i in range(0, len(self.patients))]
        self.met_list = ['Cr+PCr', 'Gln', 'Glu', 'Gly', 'GPC+PCh', 'Ins', 'NAA']
        self.patient_folders = list(pathlib.Path('data_processed').iterdir())
        self.examples = []
        self.slices = []
        self.fnames = []
        self.fname2nslice = {}
        for patient in sorted(self.patient_folders):
            patientname = str(patient.name)
            if patientname not in self.patient_list:
                continue
            met = np.load(str(patient)+'/Met_filtered/Gln.npy')
            num_slices = met.shape[0]

            self.examples += [(patient, slice, met) for slice in range(num_slices) for met in self.met_list]
            self.slices += [(str(patient), slice) for slice in range(num_slices)]
            self.fnames += [patientname]
            self.fname2nslice[patientname] = num_slices

        logging.info(' ' * 10)
        logging.info('--+' * 10)
        logging.info('loading patients: %s ' % self.fname2nslice)
        logging.info('total slices: %s' % len(self.slices))
        logging.info('total mets: %s' % len(self.examples))
        logging.info('--+' * 10)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        patient, slice, metname = self.examples[idx]
        # metabolite
        met_HR = np.load(str(patient) + '/Met_filtered/' + metname + '.npy')[slice]
        met_HR = torch.from_numpy(met_HR)
        met_max = met_HR.max()
        met_HR = met_HR / met_max

        # MRI
        T1 = np.load(str(patient)+'/MRI_sliced/T1_sliced.npy')[slice*3:(slice+1)*3, :, :]
        flair = np.load(str(patient) + '/MRI_sliced/flair_sliced.npy')[slice*3:(slice+1)*3, :, :]
        T1 = np.transpose(T1, (1, 2, 0)) / T1.max()
        flair = np.transpose(flair, (1, 2, 0)) / flair.max()

        T1, flair, met_HR = T1[None, :, :], flair[None, :, :], met_HR[None, :, :]
        #prior = {'T1': T1, 'flair': flair, 'met_max': met_max, 'patient': str(patient.name), 'slice': slice, 'metname': metname}
        prior = {'T1': T1, 'flair': flair}

        cond = {}
        if self.class_cond:
            metcode = {'Cr+PCr': 0, 'Gln': 1, 'Glu': 2, 'Gly': 3, 'GPC+PCh': 4, 'Ins': 5, 'NAA': 6}
            cond['y'] = np.array(metcode[metname], dtype=np.int64)

        return met_HR.float(), prior, met_max, str(patient.name), slice, metname, cond


def RandomDownscale_function(met_HR, lowRes):
    met_HR_cplx = torch.cat((met_HR.unsqueeze(-1), torch.zeros_like(met_HR.unsqueeze(-1))), -1)
    kspace = utils.fft2(met_HR_cplx)
    kspace_trunc = torch.zeros_like(kspace)
    if isinstance(lowRes, list):
        lowRes = torch.randint(lowRes[0], lowRes[1]+1, (1, ))[0]
    div = lowRes
    kspace_trunc[:, :, 32 - div:32 + div, 32 - div:32 + div, :] = kspace[:, :, 32 - div:32 + div, 32 - div:32 + div, :]
    met_LR_cplx = utils.ifft2(kspace_trunc)
    met_LR = torch.sqrt(met_LR_cplx[:, :, :, :, 0] ** 2 + met_LR_cplx[:, :, :, :, 1] ** 2)
    return met_LR, lowRes


def RandomFlip_function(met_HR, T1, flair):
    angle = torch.randint(0, 4, (1, ))[0]
    flip = torch.rand(1)
    if flip <= 1 / 2:
        T1 = torch.flip(torch.rot90(T1, angle, [2, 3]), [2])
        flair = torch.flip(torch.rot90(flair, angle, [2, 3]), [2])
        met_HR = torch.flip(torch.rot90(met_HR, angle, [2, 3]), [2])
    else:
        T1 = torch.rot90(T1, angle, [2, 3])
        flair = torch.rot90(flair, angle, [2, 3])
        met_HR = torch.rot90(met_HR, angle, [2, 3])

    flip_MRI = torch.rand(1)
    if flip_MRI <= 1 / 2:
        T1 = torch.flip(T1, [4])
        flair = torch.flip(flair, [4])
    return met_HR, T1, flair


def RandomShift_function(met_HR, T1, flair):
    shiftx, shifty = torch.randint(-3, 4, (1, ))[0], torch.randint(-3, 4, (1, ))[0]
    met_HR = torch.roll(met_HR, (shiftx, shifty), dims=(2, 3))
    T1 = torch.roll(T1, (shiftx * 3, shifty * 3), dims=(2, 3))
    flair = torch.roll(flair, (shiftx * 3, shifty * 3), dims=(2, 3))
    return met_HR, T1, flair


class Met_Sampler(Sampler):
    def __init__(self, batch_size, length):
        self.length = length
        self.met_list = ['Cr+PCr', 'Gln', 'Glu', 'Gly', 'GPC+PCh', 'Ins', 'NAA']
        self.num_met = len(self.met_list)
        self.indices = np.array(range(length))
        self.batch_size = batch_size

    def __iter__(self):
        batches = []
        for met_idx in range(7):
            met_indices = self.indices[0::7] + met_idx
            random.shuffle(met_indices)
            for i in range(0, math.floor(met_indices.shape[0] / self.batch_size)):
                batches.append(met_indices[i*self.batch_size:(i+1)*self.batch_size])
        random.shuffle(batches)
        #print(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.floor(self.length / 7 / self.batch_size) * 7
