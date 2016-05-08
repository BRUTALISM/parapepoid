(ns parapepoid.neural-test
  (:require [clojure.test :refer :all]
            [parapepoid.neural :refer :all]))

(defn- invertor-nn [counts training-count learning-rate]
  (let [inverse-data (fn [] (let [n (rand 1)] [[n 0] [0 n]]))]
    (train-data (network counts)
                (repeatedly training-count inverse-data)
                learning-rate)))

(defn within [a b epsilon] (< (Math/abs (- a b)) epsilon))
(defn within-all [as bs epsilon]
  (reduce #(and %1 %2) true (map within as bs (repeat epsilon))))

(deftest invertor-test
  (testing "invertor network"
    (let [epsilon 0.1
          input [0.5 0]
          in (invertor-nn [2 5 2] 100 0.2)
          output (last (feed-forward input in))]
      (is (= true (within-all (reverse output) input epsilon))))))
