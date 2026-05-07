import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
# from model_summary import get_model_flops, get_model_activation



def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    # input_dim = (3, 256, 256)  # set the input dimension
    # input_dim =torch.randn(1,3,round(720/2),round(1280/2))
    # activations, num_conv = get_model_activation(model, input_dim)
    # activations = activations / 10 ** 6
    # logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    # logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))
    #
    # input_dim =torch.randn(1,3,round(720/2),round(1280/2))
    # flops=summary(model,input_dim)
    # flops = flops / 10 ** 9
    # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    # num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # num_parameters = num_parameters / 10 ** 6
    # logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

    # ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    # logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
