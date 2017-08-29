import tensorflow as tf

thetaConvergence = .001
criticIterations = 5
batchSize = 64
learningRate = .00005
c = .01

def fLossOptimizer(f, g):
    gw = tf.divide(tf.reduce_sum(f)-tf.reduce_sum(f(g)), batchSize)
    trainable = f.getTrainableVariables() #TODO tf.trainable_variables()
    return tf.RMSPropOptimizer(learningRate).minimize(gw, var_list=trainable)

def gLossOptimizer(g):
    gTheta = -tf.divide(tf.reduce_sum(f(g)))
    trainable = g.getTrainableVariables() #TODO 
    return tf.RMSPropOptimizer(learningRate).minmize(gTheta, var_list=trainable)

def getBatch(data, batchSize): #TODO
    pass

def WGAN(f, g, xdata, zdata): 
    #TODO figure out how to store g.priorWeights, g.weights, f.weigths
    #TODO figure out how to do f(g)
    fOptimizer = fLossOptimizer(f, g)
    gOptimizer = gLossOptimizer(g)
    while tf.norm(g.priorWeights)-tf.norm(g.weights) >= thetaConvergence:
        for i in range(0,criticIterations):
            xBatch = getBatch(xdata, batchSize)
            zBatch = getBatch(zdata, batchSize)
            fOptimizer.run(feed_dict={x:xBatch, z:zBatch})
            tf.clip_by_value(f.weights, -c, c)
        zBatch = getBatch(zdata, batchSize)
        gOptimizer.run(feed_dict={z:zBatch})
