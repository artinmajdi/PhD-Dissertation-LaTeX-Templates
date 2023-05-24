import multiprocessing


from main.aims.aim1_1_taxonomy.utils_taxonomy import Visualize


# def loop():
#     for dataset_name in ['CheX', 'NIH', 'PC']:
#         for approach in ['logit', 'loss']:
#             AIM1_1_TorchXrayVision.run_full_experiment(approach=approach, dataset_name=dataset_name)

if __name__ == '__main__':

    # AIM1_1_TorchXrayVision.run_full_experiment()

    # AIM1_1_TorchXrayVision.loop_run_full_experiment()

    Visualize.loop(experiment='class_relationship', data_mode='test')

    # features, truth, _, LD = CalculateOriginalFindings.get_feature_maps(data_mode='test', dataset_name='CheX')

    print('process completed')

