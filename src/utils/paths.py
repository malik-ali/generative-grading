import os
import sys
import getpass
from glob import glob

from globals import BASE_PATH

# DATA_PATH = os.path.join(BASE_PATH, 'data')
USER = getpass.getuser()
DATA_PATH = '/mnt/fs5/{}/generative-grading/data'.format(USER)
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
STUDENT_DATA_PATH = os.path.join(DATA_PATH, 'real', 'education')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')

POS_DATA_PATH = '/mnt/fs5/{}/generative-grading/postagging/data'.format(USER)
POS_RAW_DATA_PATH = os.path.join(POS_DATA_PATH, 'raw')
POS_PROCESSED_DATA_PATH = os.path.join(POS_DATA_PATH, 'processed')

SCENE_GRAPH_DATA_PATH = '/mnt/fs5/{}/generative-grading/scenegraph/data'.format(USER)
SCENE_GRAPH_RAW_DATA_PATH = os.path.join(SCENE_GRAPH_DATA_PATH, 'raw')
SCENE_GRAPH_PROCESSED_DATA_PATH = os.path.join(SCENE_GRAPH_DATA_PATH, 'processed')


PYRAMIDS_DATA_PATH = os.path.join(BASE_PATH, 'data', 'pyramidsnapshot')
PYRAMIDS_GRAMMAR_DATA_PATH = '/mnt/fs5/{}/generative-grading/pyramidSnapshot_v2'.format(USER)

GRAMMAR_MODULE = os.path.join('src', 'rubricsampling', 'grammars')


DOMAIN_TO_RAW_DATA_PATH = {
    'education': RAW_DATA_PATH,
    'postagging': POS_RAW_DATA_PATH,
    'scenegraph': SCENE_GRAPH_RAW_DATA_PATH,
}

DOMAIN_TO_PROCESSED_DATA_PATH = {
    'education': PROCESSED_DATA_PATH,
    'postagging': POS_PROCESSED_DATA_PATH,
    'scenegraph': SCENE_GRAPH_PROCESSED_DATA_PATH,
}

def grammar_path(problem_name):
    return os.path.join(GRAMMAR_MODULE, problem_name)

def raw_data_paths(problem_name, split, domain='education', sampling_strategy='standard'):
    counts_file = 'counts_shard_*.pkl'
    labels_file = 'labels_shard_*.pkl'
    tiers_file = 'tiers_shard_*.pkl'
    rv_order_file = 'rvOrder_shard_*.pkl'
    anon_mapping_file = 'anon_mapping_shard_*.pkl'
    all_rvs_file = 'random_variables.json'

    base_dir = os.path.join(DOMAIN_TO_RAW_DATA_PATH[domain], problem_name, 
                            sampling_strategy, split)

    counts_path = os.path.join(base_dir, counts_file)
    labels_path = os.path.join(base_dir, labels_file)
    rv_order_path = os.path.join(base_dir, rv_order_file)
    tiers_path = os.path.join(base_dir, tiers_file)
    anon_mapping_path = os.path.join(base_dir, anon_mapping_file)

    all_rvs_path = os.path.join(DOMAIN_TO_RAW_DATA_PATH[domain], 
                                problem_name, all_rvs_file)

    counts_paths = sorted(glob(counts_path))
    labels_paths = sorted(glob(labels_path))
    rv_order_paths = sorted(glob(rv_order_path))
    tiers_paths = sorted(glob(tiers_path))
    anon_mapping_paths = sorted(glob(anon_mapping_path))
  
    assert len(counts_paths) == len(labels_paths)
    assert len(counts_paths) == len(anon_mapping_paths)
    assert len(counts_paths) == len(rv_order_paths)
    assert len(counts_paths) == len(tiers_paths)

    return counts_paths, labels_paths, rv_order_paths, tiers_paths, anon_mapping_paths, all_rvs_path

def raw_student_paths(problem_name):
    if problem_name != 'liftoff':
        raise ValueError('Only liftoff problem currently supported')

    base_dir = os.path.join(STUDENT_DATA_PATH, problem_name, 'raw')

    counts_file = 'liftoff_spr18_counts.pkl'
    counts_file = 'counts_spr18.pkl'
    anon_mapping_file = 'anon_mapping_spr18.pkl'
    
    data_path = os.path.join(base_dir, counts_file)
    anon_mapping_path = os.path.join(base_dir, anon_mapping_file)

    return data_path, anon_mapping_path

def raw_codeorg_student_paths(problem_name):
    base_dir = os.path.join(STUDENT_DATA_PATH, problem_name, 'raw')

    counts_file = os.path.join(base_dir, 'counts.pkl')
    labels_file = os.path.join(base_dir, 'labels.pkl')
    zipfs_file = os.path.join(base_dir, 'zipfs.pkl')
    anon_mapping_path = os.path.join(base_dir, 'anon_mapping.pkl')

    return counts_file, labels_file, zipfs_file, anon_mapping_path

def vocab_paths(problem_name, domain='education'):
    data_path = os.path.join(DOMAIN_TO_PROCESSED_DATA_PATH[domain], problem_name)

    return   {
        'data_path':                       data_path,
        'vocab_path':                      os.path.join(data_path, 'vocab.json'),
        'char_vocab_path':                 os.path.join(data_path, 'char_vocab.json'),
        'anon_vocab_path':                 os.path.join(data_path, 'anon_vocab.json'),
        'anon_char_vocab_path':            os.path.join(data_path, 'anon_char_vocab.json'),
    }

def rnn_data_paths(problem_name, split, domain='education', sampling_strategy='standard'):
    data_path = os.path.join(DOMAIN_TO_PROCESSED_DATA_PATH[domain], 
                             problem_name, sampling_strategy, split)

    student_data_path = os.path.join(STUDENT_DATA_PATH, problem_name, 'processed')                

    return {
        # ================================================================================
        #                  Overall summary data that is not shard dependent              #
        # ================================================================================
        'data_path':                       data_path,
        'student_data_path':               student_data_path,

        'rv_info_path':                    os.path.join(data_path, 'rv_info.json'),
        'metadata_path':                   os.path.join(data_path, 'metadata.json'),

        # ================================================================================
        #                                  Sharded data                                  #
        # ================================================================================

        # here we leave _shard_{} unassigned for future usage...
        # ALSO, we do not shard any vocab since we need to be consistent there.
        'raw_programs_path':               os.path.join(data_path, 'raw_programs_shard_{}.pkl'),
        'feat_programs_path':              os.path.join(data_path, 'programs_shard_{}.mat'),
        'char_feat_programs_path':         os.path.join(data_path, 'char_programs_shard_{}.mat'),

        'anon_raw_programs_path':          os.path.join(data_path, 'anon_raw_programs_shard_{}.pkl'),
        'anon_feat_programs_path':         os.path.join(data_path, 'anon_programs_shard_{}.mat'),
        'anon_char_feat_programs_path':    os.path.join(data_path, 'anon_char_programs_shard_{}.mat'),

        'feat_labels_path':                os.path.join(data_path, 'label_feats_shard_{}.npy'),
        'feat_zipfs_path':                 os.path.join(data_path, 'zipf_feats_shard_{}.npy'),
        'raw_rvOrder_path':                os.path.join(data_path, 'raw_rvOrder_shard_{}.pkl'), 
        'feat_rvOrder_path':               os.path.join(data_path, 'rvOrder_shard_{}.mat'), 


        # ================================================================================
        #                                 Test time data                                 #
        # ================================================================================ 
        # this may not be relevant (eg postagging)
        'student_programs_path':           os.path.join(student_data_path, 'student_programs.mat'),
        'student_char_programs_path':      os.path.join(student_data_path, 'student_char_programs.mat'),
        'raw_student_programs_path':       os.path.join(student_data_path, 'raw_student_programs.pkl'),

        'anon_student_programs_path':      os.path.join(student_data_path, 'anon_student_programs.mat'),
        'anon_student_char_programs_path': os.path.join(student_data_path, 'anon_student_char_programs.mat'),
        'anon_raw_student_programs_path':  os.path.join(student_data_path, 'anon_raw_student_programs.pkl')
    }


#  ---- begin small section on scene graph utilities ----


def raw_scene_graph_data_paths(problem_name, split, sampling_strategy='standard'):
    counts_file = 'counts_shard_*.pkl'
    labels_file = 'labels_shard_*.pkl'
    images_file = 'images_shard_*.pkl'
    tiers_file = 'tiers_shard_*.pkl'
    rv_order_file = 'rvOrder_shard_*.pkl'
    all_rvs_file = 'random_variables.json'

    base_dir = os.path.join(DOMAIN_TO_RAW_DATA_PATH['scenegraph'],
                            problem_name, sampling_strategy, split)

    counts_path = os.path.join(base_dir, counts_file)
    labels_path = os.path.join(base_dir, labels_file)
    images_path = os.path.join(base_dir, images_file)
    rv_order_path = os.path.join(base_dir, rv_order_file)
    tiers_path = os.path.join(base_dir, tiers_file)

    all_rvs_path = os.path.join(DOMAIN_TO_RAW_DATA_PATH['scenegraph'], 
                                problem_name, all_rvs_file)

    counts_paths = sorted(glob(counts_path))
    labels_paths = sorted(glob(labels_path))
    images_paths = sorted(glob(images_path))
    rv_order_paths = sorted(glob(rv_order_path))
    tiers_paths = sorted(glob(tiers_path))
  
    assert len(counts_paths) == len(labels_paths)
    assert len(counts_paths) == len(images_paths)
    assert len(counts_paths) == len(rv_order_paths)
    assert len(counts_paths) == len(tiers_paths)

    return (counts_paths, labels_paths, images_paths, rv_order_paths, 
            tiers_paths, all_rvs_path)



def scene_graph_data_paths(problem_name, split, sampling_strategy):
    data_path = os.path.join(DOMAIN_TO_PROCESSED_DATA_PATH['scenegraph'], 
                             problem_name, sampling_strategy, split)

    return {
        'data_path':                       data_path,
        'rv_info_path':                    os.path.join(data_path, 'rv_info.json'),
        'metadata_path':                   os.path.join(data_path, 'metadata.json'),

        'feat_images_path':                os.path.join(data_path, 'image_feats_shard_{}.mat'),

        'feat_labels_path':                os.path.join(data_path, 'label_feats_shard_{}.npy'),
        'raw_rvOrder_path':                os.path.join(data_path, 'raw_rvOrder_shard_{}.pkl'), 
        'feat_rvOrder_path':               os.path.join(data_path, 'rvOrder_shard_{}.mat'), 
    }
