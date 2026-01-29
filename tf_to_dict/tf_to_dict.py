"""
tf_to_dict.py

utility for converting a trained TensorFlow (Keras) model into a structured Python
dictionary containing layer information, model structure, and training configuration.

"""


import tensorflow as tf
import json

def tf_to_dict(tf_model):
    
    """
    convert a compiled Keras model into a streamlined dictionary containing
    key architecture details, layer parameters, and network structure.

    Parameters
    ----------
    tf_model : tf.keras.Model
        trained and compiled TensorFlow model.

    Returns
    -------
    dict
        dictionary with model metadata, layers (with weights), and connectivity info.
    """
    
    # --- Extract model architecture as a JSON dictionary ---
    tf_architecture = json.loads(tf_model.to_json())

    # --- Extract layer weights and biases ---
    tf_wb = {layer.name: layer.get_weights() for layer in tf_model.layers}
    
    print('---')
    print('wtf: converting tensorflow model to dict!')
    wtf_model = {}

    # =======================
    # find model info
    # =======================
    print('---')
    print('wtf: finding model info...')
    wtf_model['info'] = {}
    wtf_model['info']['optimiser'] = tf_architecture['compile_config']['optimizer']['class_name']
    wtf_model['info']['learning_rate'] = tf_architecture['compile_config']['optimizer']['config']['learning_rate']
    
    print(f'\tfound optimiser: {wtf_model['info']['optimiser']}')
    print(f'\tfound learning rate: {wtf_model['info']['learning_rate']}')

    # =======================
    # extract layers
    # =======================
    print('---')
    print('wtf: finding model layers...')
    wtf_model['layers'] = {}
    
    preceded_layer_names = []
    for tf_layer in tf_architecture['config']['layers']:
        wtf_layer = {}
        
        # basic layer metadata
        wtf_layer['name'] = tf_layer['name']
        wtf_layer['type'] = tf_layer['class_name']

        # input layers
        if wtf_layer['type'] == 'InputLayer':
            wtf_layer['batch_shape'] = tf_layer['config']['batch_shape']
    
            wtf_model['layers'][wtf_layer['name']] = wtf_layer
    
            print(f'\tfound {wtf_layer['name']}')

        # dense layers
        elif wtf_layer['type'] == 'Dense':
            # record inbound layer info for branching step later
            wtf_layer['inbound_layers'] = tf_layer['inbound_nodes'][0]['args'][0]['config']['keras_history'][0]
            preceded_layer_names.append(wtf_layer['name'])

            # ensure inbound shape matches build config
            assert tf_layer['build_config']['input_shape'] == tf_layer['inbound_nodes'][0]['args'][0]['config']['shape'], (
                f'\twtf: input_shape == {tf_layer['build_config']['input_shape']}, but inbound_nodes shape == {tf_layer['inbound_nodes'][0]['args'][0]['config']['shape']}' 
            )
            
            # store layer params
            wtf_layer['inbound_shape'] = tf_layer['build_config']['input_shape']
            wtf_layer['units'] = tf_layer['config']['units']
            wtf_layer['weights'] = tf_wb[tf_layer['name']][0].tolist()
            wtf_layer['biases'] = tf_wb[tf_layer['name']][1].tolist()
            wtf_layer['activation'] = tf_layer['config']['activation']

            wtf_model['layers'][wtf_layer['name']] = wtf_layer
            print(f'\tfound {wtf_layer['name']}')

        # unsupported/custom layer detected
        else:
            print('\tcustom object or layer detected! skipping - you can deal with this manually :^)')
    
    # =======================
    # interpret layer order
    # =======================
    print('---')
    print('wtf: populating dict with outbound_layers:')
    
    # find which layers follow which
    for layer_name in list(wtf_model['layers'].keys()):
        outbound_layers = []
        for preceded_layer_name in preceded_layer_names:
            if layer_name == wtf_model['layers'][preceded_layer_name]['inbound_layers']:
                outbound_layers.append(preceded_layer_name) 
        wtf_model['layers'][layer_name]['outbound_layers'] = outbound_layers

    # print detected layer order
    for layer_name in list(wtf_model['layers'].keys()):
        if len(wtf_model['layers'][layer_name]['outbound_layers']) == 0:
            print(f'\t{layer_name} has no outbound layers - output layer?')
        else:
            print(f'\t{layer_name}-->{wtf_model['layers'][layer_name]['outbound_layers'][0]}')

    # =======================
    # add structure to dict and handle branches
    # =======================
    print('---')
    print('wtf: adding detected network structure to dict:')
    
    def wtf_branch(wtf_model, layer_name, branch_layers):
        """
        function recursively trace and print network branches
        !!! these branch lists start with the branching layer !!!
        """

        # create branch entries, start with branching layer
        wtf_model['structure']['branch_0'] = {0: layer_name}
        wtf_model['structure']['branch_1'] = {0: layer_name}
        
        branch_n = 0
        for branch_layer in branch_layers[::-1]: # reverse due to weird tf ordering
            print(f'\t--- branch: {branch_n} ---')
            print(f'\t{layer_name}')
            print('\t  |')
            print('\t  v')
            
            branch_layer = str(branch_layer)
            print(f'\t{branch_layer}')
            print('\t  |')
            print('\t  v')

            # pass through branch layers until outbound detected
            next_layer = wtf_model['layers'][branch_layer]['outbound_layers']
            branch_count =1
            wtf_model['structure'][f'branch_{branch_n}'][branch_count] = branch_layer
            while len(next_layer) > 0:
                branch_count+=1
                print(f'\t{next_layer[0]}')
                wtf_model['structure'][f'branch_{branch_n}'][branch_count] = next_layer[0]
                next_layer = wtf_model['layers'][next_layer[0]]['outbound_layers']
                
                
                if len(next_layer) > 0:
                    print('\t  |')
                    print('\t  v')
                
                else:
                    print()
                
            branch_n +=1

    # entry to store structure
    wtf_model['structure'] ={}

    # trace stem until branch
    stem_count = 0
    wtf_model['structure']['stem'] = {}
    for layer_name in list(wtf_model['layers'].keys()):
        print(f'\t{layer_name}')
        wtf_model['structure']['stem'][stem_count] = layer_name

        # one outbound layer -> continue stem
        if len(wtf_model['layers'][layer_name]['outbound_layers']) == 1:
            print('\t  |')
            print('\t  v')
            stem_count += 1
            pass

        # multiple outbound layers -> branch detected
        if len(wtf_model['layers'][layer_name]['outbound_layers']) > 1:
            print('\t /\\ branch!')
            print('\tv  v')
            branch_layers = wtf_model['layers'][layer_name]['outbound_layers']
            wtf_branch(wtf_model, layer_name, branch_layers)
            break
    
    print('---')
    print('wtf: done!')

    return wtf_model