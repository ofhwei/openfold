import argparse
import ctypes
from datetime import date
import sys
import shutil


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--uniref90_database_path', type=str, default="/data/dataset/openfold/uniref90/uniref90.fasta",
    )
    parser.add_argument(
        '--mgnify_database_path', type=str, default="/data/dataset/openfold/mgnify/mgy_clusters_2018_12.fa",
    )
    parser.add_argument(
        '--pdb70_database_path', type=str, default="/home/huangwei/data/pdb70/pdb70", #最后面pdb70为pdb70_xxx的前缀，下同
    )
    parser.add_argument(
        '--uniclust30_database_path', type=str, default="/data/dataset/openfold/uniclust30/uniclust30_2018_08/uniclust30_2018_08",
    )
    parser.add_argument(
        '--bfd_database_path', type=str, default="/data/dataset/openfold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
    )
    parser.add_argument(
        '--jackhmmer_binary_path', type=str, default=shutil.which('jackhmmer'),
        help="蛋白质序列迭代搜索蛋白质序列库"
    )
    parser.add_argument(
        '--hhblits_binary_path', type=str, default=shutil.which('hhblits'),
    )
    parser.add_argument(
        '--hhsearch_binary_path', type=str, default=shutil.which('hhsearch'),
    )
    parser.add_argument(
        '--kalign_binary_path', type=str, default=shutil.which("kalign"),
    )
    parser.add_argument(
        '--max_template_date', type=str, 
        default=date.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        '--obsolete_pdbs_path', type=str, default="/data/dataset/openfold/pdb_mmcif/obsolete.dat"
    )
    parser.add_argument(
        '--release_dates_path', type=str, default=None
    )


def get_nvidia_cc():
    """
    Returns a tuple containing the Compute Capability of the first GPU
    installed in the system (formatted as a tuple of strings) and an error
    message. When the former is provided, the latter is None, and vice versa.

    Adapted from script by Jan Schlüte t
    https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """
    CUDA_SUCCESS = 0

    libnames = [
        'libcuda.so', 
        'libcuda.dylib', 
        'cuda.dll',
        '/usr/local/cuda/compat/libcuda.so', # For Docker
    ]
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        return None, "Could not load any of: " + ' '.join(libnames)

    nGpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    result = ctypes.c_int()
    device = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        if error_str.value:
            return None, error_str.value.decode()
        else:
            return None, "Unknown error: cuInit returned %d" % result
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return None, error_str.value.decode()

    if nGpus.value < 1:
        return None, "No GPUs detected"

    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return None, error_str.value.decode()

    if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != CUDA_SUCCESS:
        return None, "Compute Capability not found"

    major = cc_major.value
    minor = cc_minor.value

    return (major, minor), None
