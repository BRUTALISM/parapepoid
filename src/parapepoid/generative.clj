(ns parapepoid.generative
  (:require [parapepoid.neural :as n]
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
  (let [fns (cycle [identity m/transpose])]
    (mapv #(%1 %2) fns (reverse network))))

(defn reversed-errors
  "Calculates the difference between inputs and the result of feeding network's
  outputs back through the inverted network."
  [network inputs]
  (let [reversed (reverse-network network)
        outputs (last (n/feed-forward inputs network))
        reversed-outputs (last (n/feed-forward outputs reversed))]
    (map - reversed-outputs inputs)))

;; ↓ ↓ ↓ TESTING AREA ↓ ↓ ↓

(defn make-invertor [counts training-count learning-rate]
  (let [inverse-data (fn [] (let [n (rand 1)] [[n 0] [0 n]]))]
    (n/train-data (n/network counts)
                (repeatedly training-count inverse-data)
                learning-rate)))

(def invertor (make-invertor [2 5 2] 1000 0.2))
(def reversed-invertor (reverse-network invertor))

(n/feed-forward [0.3 0] invertor)
(n/feed-forward [-1.539573311532253E-4 0.3724641836697799] reversed-invertor)

(reversed-errors invertor [0.9 0])