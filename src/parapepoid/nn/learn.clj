(ns parapepoid.nn.learn
  (:require [clojure.core.matrix :as m]
            [parapepoid.nn.core :as n]
            [parapepoid.nn.propagation :as p]))

(defn train
  "Trains the given network by using the given batch of [inputs, outputs] pairs
  and the given learning rate."
  [network batch learning-rate]
  (let [zeroing-fn (fn [m] (m/zero-array (m/shape m)))
        start-bs (map zeroing-fn (n/biases network))
        start-ws (map zeroing-fn (n/weights network))
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
        biases (map apply-nabla (n/biases network) nabla-bs)
        weights (map apply-nabla (n/weights network) nabla-ws)]
    (n/raw-network weights biases (n/options network))))

(defn sgd
  "Trains the network using stochastic mini-batch gradient descent. Training
  data should be a list of [inputs outputs] pairs used to train the network.
  Returns the network and a list of errors calculated during training."
  [network training-data batch-size learning-rate epochs]
  (let [batch-train
        (fn [[net errors] batch]
          (let [trained (train net batch learning-rate)
                error (p/calculate-error trained batch)]
            [trained (conj errors error)]))
        train-epoch
        (fn [net-and-errors]
          (let [batches (partition batch-size batch-size []
                                   (shuffle training-data))]
            (reduce batch-train net-and-errors batches)))]
    (nth (iterate train-epoch [network []]) epochs)))
