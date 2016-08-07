(ns parapepoid.nn.core
  (:require [clojure.core.matrix :as m]
            [parapepoid.nn.activation :as a]
            [parapepoid.nn.error :as e]))

(defprotocol PNeuralNetwork
  "Defines elementary neural network operations, like getting bias and weight
  values."
  (weights [network]
    "Gets a sequence of weight matrices for all layers of the given network.")
  (biases [network]
    "Gets a sequence of bias arrays for all layers of the given network (not
    including the input \"layer\", of course).")
  (activation-fn [network]
    "Gets the activation function used to drive the given network.")
  (activation-prime-fn [network]
    "Gets the derivative of the activation function.")
  (error-fn [network]
    "Gets the error function to use when reporting training errors.")
  (error-delta-fn [network]
    "Gets the error delta function used to train the network.")
  (shape [network]
    "Gets the counts of neurons in each layer.")
  (options [network]
    "Gets the options map."))

(def activation-functions
  {:sigmoid {:main a/sigmoid
             :prime a/sigmoid-prime}})

(def error-functions
  {:quadratic {:main e/quadratic
               :delta e/quadratic-delta}
   :cross-entropy {:main e/cross-entropy
                   :delta e/cross-entropy-delta}})

(defrecord NeuralNetwork [weights biases options]
  PNeuralNetwork
  (weights [network] (:weights network))
  (biases [network] (:biases network))
  (activation-fn [network]
    (get-in activation-functions [(get-in network [:options :activation])
                                  :main]))
  (activation-prime-fn [network]
    (get-in activation-functions [(get-in network [:options :activation])
                                  :prime]))
  (error-fn [network]
    (get-in error-functions [(get-in network [:options :error-fn]) :main]))
  (error-delta-fn [network]
    (get-in error-functions [(get-in network [:options :error-fn]) :delta]))
  (shape [network]
    (into [(last (m/shape (first (:weights network))))]
          (mapv #(first (m/shape %)) (:biases network))))
  (options [network] (:options network)))

(def default-options
  {:activation :sigmoid
   :error-fn :cross-entropy})

(defn raw-network
  "Creates a neural network using already populated weight and bias matrices. If
  no options are specified, uses parapepoid.nn.core/default-options."
  ([weights biases] (raw-network weights biases default-options))
  ([weights biases options]
   (NeuralNetwork. weights biases (merge default-options options))))

(defn network
  "Makes a network with given layer sizes. The (optional) options map is used to
  configure the activation function."
  ([sizes] (network sizes default-options))
  ([sizes options]
   (let [make-weights
         (fn [[from to]]
           (let [num-elements (* from to)
                 rfn #(rand (/ 1.0 num-elements))]
             (m/array (map vec (partition from
                                          (repeatedly num-elements rfn))))))
         make-biases
         (fn [neurons]
           (m/array (repeatedly neurons #(rand (/ 1.0 neurons)))))
         weights (map make-weights (partition 2 1 sizes))
         biases (map make-biases (rest sizes))]
     (raw-network weights biases options))))