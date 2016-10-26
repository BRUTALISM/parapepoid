(ns parapepoid.nn.generative
  (:require [parapepoid.nn.core :as n]
            [parapepoid.nn.learn :as l]
            [parapepoid.nn.propagation :as p]
            [clojure.core.matrix :as m]))

(defn generate-inputs
  "Generates random inputs that are guaranteed to generate the given neural
  network outputs (within the given tolerance) when fed forward through the
  given network."
  [network outputs tolerance]
  ;; TODO: Ovo je glavna f-ja. Kad nju implementiras, zavrsio si sa ovim ns-om.
  (let []
    ))

(defn reverse-network
  "Reverses the given network, so that the input layer becomes the output, layer
  1 becomes layer n - 1, etc."
  [network]
  ;; TODO: Ovo verovatno ne moze ovako jednostavno da se resi. Istrazi.
  (n/raw-network (reverse (map m/transpose (n/weights network)))
                 (conj (vec (reverse (butlast (n/biases network))))
                       (m/new-vector (first (n/shape network))))
                 (n/options network)))

(defn reversed-errors
  "Calculates the difference between inputs and the result of feeding network's
  outputs back through the inverted network."
  [network inputs]
  (let [reversed (reverse-network network)
        outputs (p/propagate-forward network inputs)
        reversed-outputs (p/propagate-forward reversed outputs)]
    (m/sub reversed-outputs inputs)))

;; ↓ ↓ ↓ TESTING AREA ↓ ↓ ↓

(defn make-invertor [counts training-count test-count batch-size learning-rate]
  (let [inverse-data (fn [] (let [n (rand 1)] [[n 0] [0 n]]))]
    (first (l/sgd (n/network counts)
                  (repeatedly training-count inverse-data)
                  (repeatedly test-count inverse-data)
                  batch-size
                  learning-rate
                  1))))

;(def invertor (make-invertor [2 8 2] 1000 100 1 2))
;(def reversed-invertor (reverse-network invertor))
;
;(p/propagate-forward invertor [0.9 0])
;(p/propagate-forward reversed-invertor
;                     [0.016913162210121145 0.8507391898240442])

;(reversed-errors invertor [0.1 0])