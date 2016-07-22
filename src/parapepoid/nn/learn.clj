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
  [network training-data batch-size learning-rate]
  (let [batches (partition batch-size batch-size [] training-data)
        train-fn (fn [net batch] (train net batch learning-rate))]
    (reduce train-fn network batches)))

; Упореди са референтном Python имплементацијом, изгледа да нешто није у реду.
; (Не инвертује.)
(def invertor (nn/network [2 5 2]))
(defn make-invertor-training []
  (let [n (rand 1)]
    [[n 0] [0 n]]))
(def invertor-training (repeatedly 1000 make-invertor-training))
(def trained-invertor (sgd invertor invertor-training 20 1))