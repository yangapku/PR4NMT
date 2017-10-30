def setup_keyphrase_baseline():
    config = dict()
    # config['seed']            = 3030029828
    config['seed']            = 154316847
    config['task_name']       = 'baseline'
    # config['task_name']       = 'copynet-keyphrase-all.one2one.copy'
    config['timemark']        = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    config['use_noise']       = False
    config['optimizer']       = 'adam'
    config['clipnorm']        = 0.1

    config['save_updates']    = True
    config['get_instance']    = True

    config['path']            = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) #path.realpath(path.curdir)
    config['path_experiment'] = config['path'] + '/Experiment/'+config['task_name']
    config['path_h5']         = config['path_experiment']
    config['path_log']        = config['path_experiment']

    config['casestudy_log']   = config['path_experiment'] + '/case-print.log'

    # do training?
    config['do_train']        = False
    # do predicting?
    # config['do_predict']      = True
    config['do_predict']      = False
    # do testing?
    config['do_evaluate']     = True
    # config['do_evaluate']     = False
    # do validation?
    config['do_validate']     = False

    config['training_name']   = 'acm-sci-journal_600k'
    config['training_dataset']= config['path'] + '/dataset/keyphrase/million-paper/all_title_abstract_keyword_clean.json'
    config['testing_name']    = 'inspec_all'
    config['testing_dataset'] = config['path'] + '/dataset/keyphrase/inspec/inspec_all.json'

    config['testing_datasets']= ['kp20k', 'inspec', 'nus', 'semeval', 'krapivin'] # 'inspec', 'nus', 'semeval', 'krapivin', 'ke20k', 'kdd', 'www', 'umd'
    config['preprocess_type'] = 1 # 0 is old type, 1 is new type(keep most punctuation)

    config['data_process_name'] = 'eos-punctuation-1000validation/'

    config['validation_size'] = 20000
    config['validation_id']   = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'validation_id_'+str(config['validation_size'])+'.pkl'
    config['testing_id']      = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'testing_id_'+str(config['validation_size'])+'.pkl'
    config['dataset']         = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'all_600k_dataset.pkl'
    config['voc']             = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'all_600k_voc.pkl' # for manual check

    # size
    config['batch_size']      = 100
    config['mini_batch_size'] = 20
    config['mode']            = 'RNN'  # NTM
    config['binary']          = False
    config['voc_size']        = 50000

    # output log place
    if not os.path.exists(config['path_log']):
        os.mkdir(config['path_log'])

    # trained_model
    config['trained_model']   = config['path_experiment'] + '/experiments.copynet-keyphrase-all.one2one.copy.id=20161220-070035.epoch=2.batch=20000.pkl'
    # A copy-model
    # config['path_experiment'] + '/experiments.copynet-keyphrase-all.one2one.copy.id=20161220-070035.epoch=2.batch=20000.pkl'
    # A well-trained no-copy model
    # config['path_experiment'] + '/experiments.keyphrase-all.one2one.nocopy.id=20161230-000056.epoch=3.batch=1000.pkl'
    # A well-trained model on all data
    #   path.realpath(path.curdir) + '/Experiment/' + 'copynet-keyphrase-all.one2one.nocopy.<eol><digit>.emb=100.hid=150/experiments.copynet-keyphrase-all.one2one.nocopy.id=20161129-195005.epoch=2.pkl'
    # A well-trained model on acm data
    # config['path_experiment'] + '/experiments.copynet-keyphrase-all.one2one.nocopy.id=20161129-195005.epoch=2.pkl'
    config['weight_json']= config['path_experiment'] + '/model_weight.json'
    config['resume_training'] = False
    config['training_archive']= None #config['path_experiment'] + '/save_training_status.id=20161229-135001.epoch=1.batch=1000.pkl'
        #config['path_experiment'] + '/save_training_status.pkl'

    # # output hdf5 file.
    # config['weights_file']    = config['path'] + '/froslass/model-pool/'
    # if not os.path.exists(config['weights_file']):
    #     os.mkdir(config['weights_file'])

    config['max_len']         = 6
    config['sample_beam']     = config['voc_size']
    config['sample_stoch']    = False # use beamsearch
    config['sample_argmax']   = False

    config['predict_type']    = 'extractive' # type of prediction, extractive or generative
    config['predict_path']    = config['path_experiment'] + '/predict.'+config['timemark']+'.data=5.len=6.beam=all.predict=appear_only.target=appear_only/'
                                # config['path_experiment'] + '/predict.20161231-152451.len=6.beam=200.target=appear_only/'
                                # '/copynet-keyphrase-all.one2one.nocopy.extractive.predict.pkl'
    if not os.path.exists(config['predict_path']):
        os.mkdir(config['predict_path'])

    # config['path_experiment'] + '/copynet-keyphrase-all.one2one.nocopy.generate.len=8.beam=50.predict.pkl'
    # '/copynet-keyphrase-all.one2one.nocopy.extract.predict.pkl'
    #config['path_experiment'] + '/'+ config['task_name']+ '.' + config['predict_type'] + ('.len={0}.beam={1}'.format(config['max_len'], config['sample_beam'])) + '.predict.pkl' # prediction on testing data

    # Evaluation
    config['baseline_data_path']     = config['path'] + '/dataset/keyphrase/baseline-data/'

    config['normalize_score']   = True #
    # config['normalize_score']   = True
    config['predict_filter']    = 'appear-only' # [USELESS]whether do filtering on predictions? 'appear-only','non-appear-only' and None
    config['target_filter']     = None # 'appear-only' # whether do filtering on groundtruth? 'appear-only','non-appear-only' and None
    config['keep_longest']      = False # whether keep the longest phrases only, as there're too many phrases are part of other longer phrases
    config['noun_phrase_only']  = False

    config['number_to_predict'] = 10 # [desperated] the k in P@k,R@k,F1@k

    # Encoder: Model
    config['bidirectional']   = True
    config['enc_use_contxt']  = False
    config['enc_learn_nrm']   = True
    config['enc_embedd_dim']  = 150    # 100
    config['enc_hidden_dim']  = 300    # 150
    config['enc_contxt_dim']  = 0
    config['encoder']         = 'RNN'
    config['pooling']         = False

    # Decoder: dimension
    config['dec_embedd_dim']  = 150  # 100
    config['dec_hidden_dim']  = 300  # 180
    config['dec_contxt_dim']  = config['enc_hidden_dim']       \
                                if not config['bidirectional'] \
                                else 2 * config['enc_hidden_dim']

    # Decoder: CopyNet
    config['copynet']         = True
    config['identity']        = False
    config['location_embed']  = True
    config['coverage']        = True
    config['copygate']        = False

    # Decoder: Model
    config['shared_embed']    = False
    config['use_input']       = True
    config['bias_code']       = True
    config['dec_use_contxt']  = True
    config['deep_out']        = False
    config['deep_out_activ']  = 'tanh'  # maxout2
    config['bigram_predict']  = True
    config['context_predict'] = True
    config['dropout']         = 0.5  # 5
    config['leaky_predict']   = False

    config['dec_readout_dim'] = config['dec_hidden_dim']
    if config['dec_use_contxt']:
        config['dec_readout_dim'] += config['dec_contxt_dim']
    if config['bigram_predict']:
        config['dec_readout_dim'] += config['dec_embedd_dim']


    # Decoder: sampling
    config['multi_output']    = False

    config['decode_unk']      = False
    config['explicit_loc']    = False

    # Gradient Tracking !!!
    config['gradient_check']  = True
    config['gradient_noise']  = True

    config['skip_size']       = 15

    # for w in config:
    #     print '{0} => {1}'.format(w, config[w])
    # print 'setup ok.'
    return config
