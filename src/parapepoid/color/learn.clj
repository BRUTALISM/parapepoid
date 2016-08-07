(ns parapepoid.color.learn
  (:require [parapepoid.util :as u]
            [parapepoid.color.core :as c]
            [parapepoid.nn.core :as n]
            [parapepoid.nn.learn :as l]
            [parapepoid.nn.propagation :as p]
            [parapepoid.serialization :as s]
            [thi.ng.color.core :as tc]))

(def flatten-hsl
  "A transducer for unpacking a sequence of colors into a sequence of floating
  point numbers representing HSL values."
  (let [to-hsl (map tc/as-hsla)
        unpack (mapcat #(u/select-values % [:h :s :l]))]
    (comp to-hsl unpack)))

(defn read-data
  "Reads the training data from the given file and splits it in two parts,
  training and test, which are returned inside a vector."
  [filename]
  (let [data (s/read-training filename)
        prepare (map #(vector (into [] flatten-hsl (first %1))
                              (vector (second %1))))]
    (into [] prepare data)))

(defn evaluate-hyper-params
  "High-level function used for evaluating the given set of hyper-parameters for
  the given training and test data. A network is created using parameters in the
  params map (explained below), it is trained using the training data, and then
  the error on test data is calculated and returned.

  The parameters you should specify in the params map are:
    :hidden-sizes - a vector of integers representing the neuron count for each
      hidden layer)
    :learning-rate - the learning rate used when learning from training data
    :batch-size - the batch size used during stochastic gradient descent
    :epochs - how many times the training data is used to retrain the network
    :error-fn - which error function to use (note that this also controls which
      error delta function will be used during learning)"
  [training-data test-data params]
  (let [{:keys [hidden-sizes learning-rate batch-size epochs error-fn]} params
        input-count (count (first (first training-data)))
        network (n/network (concat [input-count] hidden-sizes [1])
                           {:error-fn error-fn})
        trained-network (l/sgd network training-data batch-size learning-rate
                               epochs)
        error (p/calculate-error trained-network test-data)]
    error))

(defn generate-hyper-params
  [source-params]
  (let []
    ; TODO: Implement.
    source-params))

(defn iterate-hyper
  [data-file test-percentage iterations]
  (let [all-data (read-data data-file)
        training-count (* (- 1.0 test-percentage) (count all-data))
        iteratefn
        (fn []
          (let [shuffled (shuffle all-data)
                [training-data test-data] (split-at training-count shuffled)
                hyper-params (generate-hyper-params {:hidden-sizes [4]
                                                     :learning-rate 0.5
                                                     :batch-size 20
                                                     :epochs 1
                                                     :error-fn :cross-entropy})]
            [hyper-params
             (evaluate-hyper-params training-data test-data hyper-params)]))]
    (repeatedly iterations iteratefn)))

(iterate-hyper "TR-I3-O1-RAND.clj" 0.2 3)