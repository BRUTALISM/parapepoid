(ns parapepoid.nn.learn
  (:require [clojure.core.matrix :as m]
            [parapepoid.nn.core :as nn]
            [parapepoid.nn.propagation :as p]))

(defn train
  "Trains the given network by using the given batch of [inputs, outputs] pairs
  and the given learning rate."
  [network batch learning-rate]
  (let [zeroing-fn (fn [m] (m/zero-array (m/shape m)))
        start-bs (map zeroing-fn (nn/biases network))
        start-ws (map zeroing-fn (nn/weights network))
        single-batch
        (fn [[bs ws] [ins outs]]
          (let [nablas (p/propagate-backward network ins outs)]
            [(map m/add bs (:nabla-biases nablas))
             (map m/add ws (:nabla-weights nablas))]))
        [nabla-bs nabla-ws] (reduce single-batch [start-bs start-ws] batch)
        nabla-scale-factor (/ learning-rate (count batch))
        apply-nabla
        (fn [m nabla]
          (m/sub m (m/mul nabla nabla-scale-factor)))
        biases (map apply-nabla (nn/biases network) nabla-bs)
        weights (map apply-nabla (nn/weights network) nabla-ws)]
    (nn/raw-network weights biases)))

(defn sgd
  "Trains the network using stochastic mini-batch gradient descent. Training
  data should be a list of [inputs outputs] pairs used to train the network."
  [network training-data batch-size learning-rate epochs]
  (let [batch-train (fn [net batch] (train net batch learning-rate))
        train-epoch
        (fn [net]
          (let [batches (partition batch-size batch-size []
                                   (shuffle training-data))]
            (reduce batch-train net batches)))]
    (nth (iterate train-epoch network) epochs)))
