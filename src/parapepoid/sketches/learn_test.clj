(ns parapepoid.sketches.learn-test
  (:require [parapepoid.nn.core :as n]
            [parapepoid.nn.learn :as l]
            [parapepoid.nn.propagation :as p]
            [incanter.core :as inc]
            [incanter.charts :as chart]))

(def network (n/network [3 20 1]))

(def yes [1.0 0.0 0.0])
(defn yes-training []
  [yes [1.0]])

(defn random-triplet [] (vec (repeatedly 3 rand)))
(defn no-training []
  [(random-triplet) [0.0]])

(defn training [ycount ncount]
  (-> []
      (into (take ycount (repeatedly yes-training)))
      (into (take ncount (repeatedly no-training)))))

(def tr (training 10 10))
(def sgd-result (l/sgd network tr tr 1 0.2 100))
(def learned (first sgd-result))
(def errors (second sgd-result))

(inc/view (chart/xy-plot (range (count errors)) errors
                         :x-label "Epoch"
                         :y-label "Test error (red color classifier)"))

;(defn train-test [net]
;  (let [yesprop (p/propagate-forward net yes)
;        random-no (random-triplet)
;        noprop (p/propagate-forward net random-no)]
;    (println yes " → " yesprop)
;    (println random-no " → " noprop)))
;
;(train-test learned)

; The nn implementation is correct. The above example works, so it's probably
; that the TR-I3-O1-RAND.clj training data is just too random for the network
; to infer any meaningful patterns. I tried upping the number of hidden neurons
; to 1000 and increasing the number of epochs to 1000+, and only then has the
; network started to exhibit non-uniform outputs, but the error was still going
; wild.
; So, try making a more focused color generator - dark shades of blue and red,
; for example – and see if it works.