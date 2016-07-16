(ns parapepoid.nn.propagation
  (:require [clojure.core.matrix :as m]
            [parapepoid.nn.core :as nn]))

(defn propagate-forward
  "Returns the outputs of the network when inputs are fed into it."
  [network inputs]
  (let [activation-fn (nn/activation-fn network)
        weights (nn/weights network)
        biases (nn/biases network)
        rfn (fn [in [ws bs]] (m/emap activation-fn (m/add (m/mmul ws in) bs)))]
    (reduce rfn inputs (partition 2 (interleave weights biases)))))