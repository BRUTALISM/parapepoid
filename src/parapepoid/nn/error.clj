(ns parapepoid.nn.error
  (:require [clojure.core.matrix :as m]))

(defn quadratic
  "Calculates the quadratic error for the given output layer activations (as)
  and target outputs (ys)."
  [as ys]
  (/ (m/magnitude-squared (m/sub as ys)) 2))

(defn quadratic-delta
  "Calculates the quadratic error delta for the given weighted input sums (zs),
  activations (as), target outputs (ys), and the derivative of the activation
  function (apfn)."
  [zs as ys apfn]
  (m/mul (m/sub as ys)
         (m/emap apfn zs)))

(defn cross-entropy
  "Calculates the cross-entropy error for the given output layer activations
  (as) and target outputs (ys)."
  [as ys]
  (let [log (fn [x] (Math/log x))
        nan-to-zero (fn [x]
                      (if (or (Double/isNaN x) (Double/isInfinite x)) 0 x))
        first-term (m/mul (m/sub ys 1) (m/emap log (m/sub 1 as)))
        second-term (m/mul ys (m/emap log as))
        vector-error (m/sub first-term second-term)]
    (m/esum (m/emap nan-to-zero vector-error))))

(defn cross-entropy-delta
  "Calculates the cross-entropy error delta for the given activations (as) and
  target outputs (ys)."
  [_ as ys _]
  (m/sub as ys))