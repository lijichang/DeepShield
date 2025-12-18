

DATASET_CONFIG = {
    'ffpp': {
        'num_classes': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'test.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'ffpp_combine': {
        'num_classes': 2,
        'train_list_name': 'combine_train.txt',
        'val_list_name': 'test.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'ffpp_df_com': {
        'num_classes': 2,
        'train_list_name': 'combine_train_df.txt',
        'val_list_name': 'test_df.txt',
        'test_list_name': 'test_df.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'ffpp_f2f_com': {
        'num_classes': 2,
        'train_list_name': 'combine_train_f2f.txt',
        'val_list_name': 'test_f2f.txt',
        'test_list_name': 'test_f2f.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'ffpp_fs_com': {
        'num_classes': 2,
        'train_list_name': 'combine_train_fs.txt',
        'val_list_name': 'test_fs.txt',
        'test_list_name': 'test_fs.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'ffpp_nt_com': {
        'num_classes': 2,
        'train_list_name': 'combine_train_nt.txt',
        'val_list_name': 'test_nt.txt',
        'test_list_name': 'test_nt.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'cdfv2': {
        'num_classes': 2,
        'train_list_name': 'celeb_test.txt',
        'val_list_name': 'celeb_test.txt',
        'test_list_name': 'celeb_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'dfdc': {
        'num_classes': 2,
        'train_list_name': 'dfdc_test.txt',
        'val_list_name': 'dfdc_test.txt',
        'test_list_name': 'dfdc_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'dfdcp': {
        'num_classes': 2,
        'train_list_name': 'dfdcp_test.txt',
        'val_list_name': 'dfdcp_test.txt',
        'test_list_name': 'dfdcp_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    },
    'dfd': {
        'num_classes': 2,
        'train_list_name': 'dfd_test.txt',
        'val_list_name': 'dfd_test.txt',
        'test_list_name': 'dfd_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3,
    }
}


def get_dataset_config(dataset, use_lmdb=False):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['train_list_name']
    val_list_name = ret['val_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['val_list_name']
    test_list_name = ret['test_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['test_list_name']
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
