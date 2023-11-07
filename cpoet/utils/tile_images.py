from PIL import Image 


image_filenames = [
    # '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/baseline_wCM_14nov_seed24582923/cp-1500/coverage_metric_N1000/score_vs_k_n1000.png',
    '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/baseline_wCM_17nov_seed24582924/[cp-1500]->cp-1600/coverage_metric_N1000/score_vs_k_n1000.png',
    # '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17nov/cp-1500/coverage_metric_N1000/score_vs_k_n1000.png',

    '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_19nov/cp-1500/coverage_metric_N1000/score_vs_k_n1000.png',
    '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8nov/[cp-1300]->cp-1500/coverage_metric_N1000/score_vs_k_n1000.png',
    '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma20.0_wCM_8nov/cp-1500/coverage_metric_N1000/score_vs_k_n1000.png',
    '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_16nov/cp-1500/coverage_metric_N1000/score_vs_k_n1000.png',


    # '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_19nov/[cp-1800]->cp-1900/coverage_metric_N1000/score_vs_k_n1000.png',
    # '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8nov/[[cp-1300]->cp-1550]->cp-2000/coverage_metric_N1000/score_vs_k_n1000.png',
    # '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma20.0_wCM_8nov/cp-2000/coverage_metric_N1000/score_vs_k_n1000.png',
    # '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_16nov/cp-2000/coverage_metric_N1000/score_vs_k_n1000.png',
    # '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_16nov/cp-300/coverage_metric_N1000/score_vs_k_n1000.png',

]

# dest_filename = '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/Baselines_14_17_0_iter_1500.png'
dest_filename = '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/ICM_nov17_5_10_20_30_iter_1500.png'


temp = Image.open(image_filenames[0])
conjoined = Image.new('RGB', (temp.width*len(image_filenames), temp.height)) 

for index, fn in enumerate(image_filenames):
    im = Image.open(fn)
    conjoined.paste(im, (index * temp.width, 0))


conjoined.save(dest_filename)