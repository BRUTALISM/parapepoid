(ns parapepoid.nn.propagation
  (:require [clojure.core.matrix :as m]
            [parapepoid.nn.core :as n]))

(defn propagate-forward
  "Returns the outputs of the network when inputs are fed into it."
  [network inputs]
  (let [activation-fn (n/activation-fn network)
        weights (n/weights network)
        biases (n/biases network)
        rfn (fn [in [ws bs]] (m/emap activation-fn (m/add (m/mmul ws in) bs)))]
    (reduce rfn inputs (map vector weights biases))))

(defn propagate-backward
  "Propagates the given inputs forward, then calculates the per-layer bias and
  weight gradients and returns them inside a map with :nabla-biases and
  :nabla-weights keys, respectively."
  [network inputs targets]
  ; TODO: Transducerssssss!!
  (let [activation (n/activation-fn network)
        activation-prime (n/activation-prime-fn network)
        ws (n/weights network)
        bs (n/biases network)
        forward-fn
        (fn [[in-as in-zs] [ws bs]]
          (let [zs (m/add (m/mmul ws (last in-as)) bs)
                as (m/emap activation zs)]
            [(conj in-as as) (conj in-zs zs)]))
        [as zs] (reduce forward-fn
                        [[inputs] []]
                        (map vector ws bs))
        error-delta-fn (n/error-delta-fn network)
        delta (error-delta-fn (last zs) (last as) targets activation-prime)
        nabla-bias-fn
        (fn [nabla-biases-so-far [weights zs]]
          (conj nabla-biases-so-far
                (m/mul (m/mmul (m/transpose weights)
                               (first nabla-biases-so-far))
                       (m/emap activation-prime zs))))
        nabla-bias-data (map vector (reverse ws) (reverse (butlast zs)))
        nabla-biases (reduce nabla-bias-fn (list delta) nabla-bias-data)
        matrixized-biases (map #(m/array (mapv vector %1)) nabla-biases)
        nabla-weights (map #(m/mmul %1 (m/array [%2]))
                           matrixized-biases (butlast as))]
    {:nabla-biases nabla-biases
     :nabla-weights nabla-weights}))

(defn calculate-error
  "Returns the total error for the given set of [input, output] pairs in
  test-data."
  [network test-data]
  ; TODO: Add regularization support.
  (let [error-fn (n/error-fn network)
        data-length (count test-data)
        iterate-error
        (fn [error [inputs outputs]]
          (+ error
             (/ (error-fn (propagate-forward network inputs) outputs)
                data-length)))]
    (reduce iterate-error 0.0 test-data)))