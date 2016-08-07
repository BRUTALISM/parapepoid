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
  [filename test-percentage]
  (let [data (s/read-training filename)
        prepare (map #(vector (into [] flatten-hsl (first %1))
                              (vector (second %1))))
        prepared-data (into [] prepare data)
        training-count (* (- 1.0 test-percentage) (count prepared-data))]
    (split-at training-count prepared-data)))

(defn evaluate-hyper-params
  "High-level function used for evaluating the given set of hyper-parameters for
  the given training data. [inputs outputs] pairs are read from the specified
  file and the data is split into training and test data using the
  :test-data-percentage value in the params map. A network is created using the
  other parameters in the params map (explained below), the network is trained
  using training data, and then the error on test data is calculated and
  returned.

  The parameters you can specify in the params map are:
    :hidden-sizes - a vector of integers representing the neuron count for each
      hidden layer)
    :learning-rate - the learning rate used when learning from training data
    :batch-size - the batch size used during stochastic gradient descent
    :epochs - how many times the training data is used to retrain the network
    :error-fn - which error function to use (note that this also controls which
      error delta function will be used during learning)"
  [data-file params]
  (let [{:keys [test-data-percentage hidden-sizes learning-rate batch-size
                epochs error-fn]} params
        [training-data test-data] (read-data data-file test-data-percentage)
        input-count (count (first (first training-data)))
        network (n/network (concat [input-count] hidden-sizes [1])
                           {:error-fn error-fn})
        trained-network (l/sgd network training-data batch-size learning-rate
                               epochs)
        error (p/calculate-error trained-network test-data)]
    error))

(evaluate-hyper-params "TR-I3-O1-RAND.clj"
                       {:test-data-percentage 0.2
                        :hidden-sizes [4]
                        :learning-rate 0.5
                        :batch-size 20
                        :epochs 1
                        :error-fn :cross-entropy})

; TODO: pronadji najbolje meta-parametre za zadate trening podatke