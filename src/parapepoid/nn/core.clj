(ns parapepoid.nn.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mo]
            [parapepoid.nn.activation :as a]))

(defprotocol PNeuralNetwork
  "Defines elementary neural network operations, like getting bias and weight
  values."
  (weights [this]
    "Gets a sequence of weight matrices for all layers of the given network.")
  (biases [this]
    "Gets a sequence of bias arrays for all layers of the given network (not
    including the input \"layer\", of course).")
  (activation-fn [this]
    "Gets the activation function used to drive the given network.")
  (activation-prime-fn [this]
    "Gets the derivative of the activation function."))

(def activation-functions
  {:sigmoid {:main a/sigmoid
             :prime a/sigmoid-prime}})

(defrecord NeuralNetwork [weights biases options]
  PNeuralNetwork
  (weights [this] (:weights this))
  (biases [this] (:biases this))
  (activation-fn [this] (-> activation-functions
                            (-> this :options :activation) :main))
  (activation-prime-fn [this] (-> activation-functions
                                  (-> this :options :activation) :prime)))

(defn network
  "Makes a network with given layer sizes."
  ([sizes] (network sizes {:activation :sigmoid}))
  ([sizes options]
   (let [make-weights
         (fn [[from to]]
           (let [num-elements (* from to)
                 rfn #(rand (/ 1.0 num-elements))]
             (m/array (map vec (partition to (repeatedly num-elements rfn))))))
         make-biases
         (fn [neurons]
           (m/array (repeatedly neurons #(rand (/ 1.0 neurons)))))
         weights (map make-weights (partition 2 1 sizes))
         biases (map make-biases (rest sizes))]
     (NeuralNetwork. weights biases options))))