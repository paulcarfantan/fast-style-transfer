from __future__ import print_function
import functools
import vgg
import random
#import pdb
import time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):                      
    
    #content_targets = liste des éléments du dossier --train-path ? ? ? ? ?
    #style_target = array à 3 dimensions
    #batch_size = nombre de samples qui vont traverser le neural network en même temps
         
    if slow:
        batch_size = 1
    
    mod = len(content_targets) % batch_size
    if mod > 0:                                           #On veut un nb d'éléments qui soit un multiple de batch size            
        print("Train set has been trimmed slightly..")    #=> les derniers samples sont supprimés pour ne pas avoir de batch (lot) incomplet
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size,256,256,3)      #4 dimensions
    style_shape = (1,) + style_target.shape   #4 dimensions
    print(style_shape)

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)                                   #style_image - mean_pixel
        net = vgg.net(vgg_path, style_image_pre)                                        #idem neural_style
        style_pre = np.array([style_target])
        
        for layer in STYLE_LAYERS:                                                      #('relu*_1')     *: 1-5                           
            features = net[layer].eval(feed_dict={style_image:style_pre})               #évalue pour chaque style layer net[layer] avec le dictionnaire : {key=placeholder , value=tensor}
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram                                       #style_features = liste des matrices de gram de chaque style_layer  (=> 5 éléments)


    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]    #CONTENT_LAYER : 1 élément ('relu4_2')

        #preds_pre
        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        #CONTENT LOSS = idem neural_style
        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )
        #net[CONTENT_LAYER] = features de l'image générée ; content_features[CONTENT_LAYER] = features de l'image d'origine 
        
        #STYLE LOSS = idem neural_style
        style_losses = []
        for style_layer in STYLE_LAYERS:        #STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            #gram = style representation of generated image ; style_gram = style representation of original image 
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size
        #          = style_weight * somme(style_loss de chaque layer) / batch_size

        #TV LOSS = idem neural_style
        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size


        # overall loss
        loss = content_loss + style_loss + tv_loss

      
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)      #idem neural_style, mais sans les paramètres beta1, beta2, epsilon
        
        sess.run(tf.global_variables_initializer())
        
#        import random
        uid = random.randint(1, 100)           #UID : entier aléatoire entre 1 et 100
        print("UID: %s" % uid)
        
        for epoch in range(epochs):               # default epochs = 2
            num_examples = len(content_targets)   # = nombre de fichier dans le dossier --train-path
            iterations = 0
            while iterations * batch_size < num_examples:     # <=> for i in range(nombre de batch à faire passer dans le neural network)
                start_time = time.time()                      # Initialisation du temps
                curr = iterations * batch_size                # = nb de samples ayant traversé
                step = curr + batch_size                      # = nb de samples qui auront traversé après cette itération
                X_batch = np.zeros(batch_shape, dtype=np.float32)              # batch_shape = ( batch_size , 256 , 256 , 3 )
                
                for j, img_p in enumerate(content_targets[curr:step]):         # pour toutes les images du batch en cours
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)        # on convertit en array les images, puis on les resize (256,256,3)

                iterations += 1                      # incrementation
                assert X_batch.shape[0] == batch_size         

                feed_dict = {
                   X_content:X_batch
                }

                train_step.run(feed_dict=feed_dict)  # on exécute l'optimizer ; et on remplace les X_batch dans le feed_dict par les input optimisés
                
                end_time = time.time()                          # Fin du temps
                delta_time = end_time - start_time              # On calcule le temps nécessaire pour 1 batch de ...
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                
                is_print_iter = int(iterations) % print_iterations == 0    # moment de print-iterations
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples    # fin du pour ET fin du while
                should_print = is_print_iter or is_last
                
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]     # fetches (éléments que sess.run() doit aller chercher)
                    test_feed_dict = {
                       X_content:X_batch
                    }                          # X_content = placeholder  ;  X_batch = tensor
                    tup = sess.run(to_get, feed_dict = test_feed_dict)         # on exécute une étape du tensorflow en réalisant les opérations décrites, puis on évalue tous les fetches
                                                                               # les fetches remplacent les values de test_feed_dict
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup      # ~ évaluation des tensors
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)                   # si should_print, alors return(...) ; sinon, rien !

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
