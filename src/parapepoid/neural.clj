(ns parapepoid.neural
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mo]))

(defn generate-weights
  "Generates a matrix of weights needed to connect a layer with from neurons to
  a layer with to neurons."
  [from to]
  (let [l (* from to)]
    (map vec (partition to (repeatedly l #(rand (/ 1 l)))))))

(defn network
  "Creates a neural network with the number of neurons in each layer given by
  elements in the counts collection. For example, (network [2 3 1]) creates a
  network with two input neurons, one hidden layer with three neurons, and one
  neuron in the output layer; (network [15 10 8 4]) creates a network with two
  hidden layers (with 10 and 8 neurons, respectively), and so on."
  [counts]
  (let [lfn (fn [[l1 l2]]
              (mapv vec [(repeat l1 0.0) (generate-weights l1 l2)]))
        lefts (vec (mapcat lfn (partition 2 1 counts)))]
    (conj lefts (vec (repeat (last counts) 0.0)))))

(defn activation [x] (Math/tanh x))

(defn dactivation [x] (- 1.0 (* x x)))

(defn layer-activation [inputs strengths]
  (mapv activation (mapv #(reduce + %) (m/mul inputs (m/transpose strengths)))))

(defn output-deltas [targets outputs]
  (m/mul (mapv dactivation outputs)
         (mo/- targets outputs)))

(defn hidden-deltas [odeltas neurons strengths]
  (m/mul (mapv dactivation neurons)
         (mapv #(reduce + %) (m/mul odeltas strengths))))

(defn update-strengths [deltas neurons strengths lrate]
  (m/add strengths
         (m/mul lrate (mapv #(m/mul deltas %) neurons))))

(defn feed-forward
  "Feeds forward the input through the given network and returns the network
  with updated layers."
  [input network]
  (let [layers (take-nth 2 (assoc network 0 input))
        weights (take-nth 2 (rest network))
        rfn (fn [ls ws] (conj ls (layer-activation (last ls) ws)))
        updated-layers (vec (reduce rfn [input] weights))]
    (into [(first updated-layers)] (interleave weights (rest updated-layers)))))

(defn update-weights
  "Updates all weights in the given network by calculating the error of its
  output neurons compared to the given target values and backpropagating the
  differences scaled using the given learning rate lr."
  [network target lr]
  (let [blocks (partition 2 network) ;; (layer weights) pairs
        odeltas (output-deltas target (last network))
        deltafn (fn [deltas block]
                  (let [[from from->to] block
                        delta (last deltas)]
                    (conj deltas (hidden-deltas delta from from->to))))
        all-deltas (reverse (reduce deltafn [odeltas] (reverse (rest blocks))))
        weightfn (fn [[deltas [from from->to]]]
                   (update-strengths deltas from from->to lr))
        all-weights (map weightfn (map vector all-deltas blocks))]
    (conj (vec (mapcat vector (map first blocks) all-weights))
          (last network))))

(defn train
  "Trains the given network using the given input and target neuron values and
  learning rate lr. The network is forward-propagated to calculate its output
  neuron values which are then used against the target values to compute the
  error. The error is then fed backwards through the network using the given
  learning rate in order to adjust the network's weights."
  [network input target lr]
  ;; TODO: train treba da prima *listu* inputa i *listu* targeta i da racuna
  ;; gresku kumulativno nad svim primerima iz trening skupa (kao po definiciji
  ;; sa Coursera-e ili iz one knjige). Trenutno ide jedan po jedan. Radi, ali
  ;; radice bolje ako bude po definiciji.
  (update-weights (feed-forward input network) target lr))

(defn train-data
  "Trains the network by iteratively feeding it [input target] pairs read from
  the data collection, using the given learning rate lr."
  [network data lr]
  (if-let [[input target] (first data)]
    (recur (train network input target lr) (rest data) lr)
    network))
