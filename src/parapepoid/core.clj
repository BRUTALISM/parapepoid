(ns parapepoid.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mo]))

(m/set-current-implementation :vectorz)

;; --- Network definition

(def learning-rate 0.2)
(def input-neurons (m/array [1 0]))
(def input->hidden (m/array [[0.12 0.2 0.13]
                             [0.01 0.02 0.03]]))

(def hidden-neurons (m/array [0 0 0]))
(def hidden->output (m/array [[0.15 0.16]
                              [0.02 0.03]
                              [0.01 0.02]]))

(def targets (m/array [0 1]))

(defn activation [x] (Math/tanh x))
(defn dactivation [x] (- 1.0 (* x x)))

;; --- Forward propagation

(defn layer-activation [inputs strengths]
  (mapv activation (mapv #(reduce + %) (m/mul inputs (m/transpose strengths)))))

(def new-hidden-neurons (layer-activation input-neurons input->hidden))
(def new-output-neurons (layer-activation new-hidden-neurons hidden->output))

;; --- Backward propagation (error calculation)

(defn output-deltas [targets outputs]
  (m/mul (mapv dactivation outputs)
         (mo/- targets outputs)))

(def odeltas (output-deltas targets outputs))

(defn hidden-deltas [odeltas neurons strengths]
  (m/mul (mapv dactivation neurons)
         (mapv #(reduce + %) (m/mul odeltas strengths))))

(def hdeltas (hidden-deltas odeltas new-hidden-neurons hidden->output))

(defn update-strengths [deltas neurons strengths lrate]
  (m/add strengths
         (m/mul lrate
                (mapv #(m/mul deltas %) neurons))))

(def new-hidden->output
  (update-strengths odeltas new-hidden-neurons hidden->output learning-rate))
(def new-input->hidden
  (update-strengths hdeltas input-neurons input->hidden learning-rate))

;; --- Generalized representation

(def nn [[1 0]
         input->hidden
         hidden-neurons
         hidden->output
         [0 0]])

(defn feed-forward [input network]
  (let [[input input->hidden hidden hidden->output output] network
        new-hidden (layer-activation input input->hidden)
        new-output (layer-activation new-hidden hidden->output)]
    [input input->hidden new-hidden hidden->output new-output]))

;; [input-neurons input->hidden new-hidden-neurons hidden->output new-output-neurons]
;; (feed-forward (m/array [1 0]) nn)

(defn update-weights [network target learning-rate]
  (let [[input input->hidden hidden hidden->output output] network
        odeltas (output-deltas target output)
        hdeltas (hidden-deltas odeltas hidden hidden->output)
        new-hidden->output (update-strengths odeltas
                                             hidden
                                             hidden->output
                                             learning-rate)
        new-input->hidden (update-strengths hdeltas
                                            input
                                            input->hidden
                                            learning-rate)]
    [input new-input->hidden hidden new-hidden->output output]))

;; [input-neurons new-input->hidden new-hidden-neurons new-hidden->output new-output-neurons]
;; (update-weights (feed-forward [1 0] nn) [0 1] learning-rate)

(defn train-network [network input target learning-rate]
  (update-weights (feed-forward input network) target learning-rate))

;; --- Training data

(defn train-data [network data learning-rate]
  (if-let [[input target] (first data)]
    (recur
     (train-network network input target learning-rate)
     (rest data)
     learning-rate)
    network))

(defn ff [input network]
  (last (feed-forward input network)))

(ff [1 0] nn)

(def n1 (train-data nn [[[1 0] [0 1]]
                        [[0.5 0] [0 0.5]]
                        [[0.25 0] [0 0.25]]] 0.5))
(ff [1 0] n1)

(defn inverse-data []
  (let [n (rand 1)]
    [[n 0] [0 n]]))

;; --- General construct network

(defn gen-strengths [to from]
  (let [l (* to from)]
    (map vec (partition from (repeatedly l #(rand (/ 1 l)))))))

(defn construct-network [num-in num-hidden num-out]
  (vec (map vec [(repeat num-in 0)
                 (gen-strengths num-in num-hidden)
                 (repeat num-hidden 0)
                 (gen-strengths num-hidden num-out)
                 (repeat num-out 0)])))

;; --- Trying it all out

(defn invertor-nn [num-hidden training-count learning-rate]
  (train-data (construct-network 2 num-hidden 2)
              (repeatedly training-count inverse-data)
              learning-rate))

(def nn4 (invertor-nn 4 1000 0.5))
nn4
(feed-forward [0.5 0] nn4)

;; --- Other

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
