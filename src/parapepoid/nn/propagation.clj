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
    (reduce rfn inputs (map vector weights biases))))

(defn quadratic-delta
  "Returns the quadratic cost delta for the given weighted input sums (zs),
  activations (as), target outputs (ys), and the derivative of the activation
  function (apfn)."
  [zs as ys apfn]
  (m/mul (m/sub as ys)
         (m/emap apfn zs)))

(defn cross-entropy-delta
  "Returns the cross-entropy cost delta for the given activations (as) and
  target outputs (ys)."
  [as ys]
  (m/sub as ys))

(defn propagate-backward
  "Propagates the given inputs forward, then calculates the per-layer bias and
  weight gradients and returns them inside a map with :nabla-biases and
  :nabla-weights keys, respectively."
  [network inputs targets]
  ; TODO: Transducerssssss!!
  (let [activation (nn/activation-fn network)
        activation-prime (nn/activation-prime-fn network)
        ws (nn/weights network)
        bs (nn/biases network)
        forward-fn
        (fn [[in-as in-zs] [ws bs]]
          (let [zs (m/add (m/mmul ws (last in-as)) bs)
                as (m/emap activation zs)]
            [(conj in-as as) (conj in-zs zs)]))
        [as zs] (reduce forward-fn
                        [[inputs] []]
                        (map vector ws bs))
        ;delta (quadratic-delta (last zs) (last as) targets activation-prime)
        delta (cross-entropy-delta (last as) targets)
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
