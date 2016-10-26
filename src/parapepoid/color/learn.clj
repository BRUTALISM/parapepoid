(ns parapepoid.color.learn
  (:require [parapepoid.util :as u]
            [parapepoid.color.core :as c]
            [parapepoid.nn.core :as n]
            [parapepoid.nn.learn :as l]
            [parapepoid.nn.propagation :as p]
            [parapepoid.serialization :as s]
            [parapepoid.color.serialization :as cs]
            [parapepoid.approach.core :as a]
            [parapepoid.approach.input-mappings :as ai]
            [parapepoid.approach.palette-generators :as ag]))

(def error-fn :cross-entropy)

(def extents
  {:hidden-sizes [1 50]
   :learning-rate [0.00001 500.0]
   :batch-size [1 500]
   :epochs [1 20]})

(defn evaluate-hyper-params
  "High-level function used for evaluating the given set of hyper-parameters for
  the given training and test data. A network is created using parameters in the
  params map (explained below), and it is trained using the training data.
  Returns the network and the errors on test data per epoch.

  The parameters you should specify in the params map are:
    :hidden-sizes - a vector of integers representing the neuron count for each
      hidden layer
    :learning-rate - the learning rate used when learning from training data
    :batch-size - the batch size used during stochastic gradient descent
    :epochs - how many times the training data is used to retrain the network
    :error-fn - which error function to use (note that this also controls which
      error delta function will be used during learning)"
  [training-data test-data params]
  (let [{:keys [hidden-sizes learning-rate batch-size epochs error-fn]} params
        input-count (count (first (first training-data)))
        network (n/network (concat [input-count] hidden-sizes [1])
                           {:error-fn error-fn})]
    (l/sgd network training-data test-data batch-size learning-rate epochs)))

(defn generate-hyper-params
  "Generates a new hyper-parameters map based on the given source-params map, if
  given; otherwise generates a random map while obeying parameter extents
  defined in the top-level extents map in this namespace."
  ([]
   (let [random-extent
         (fn [key int?]
           (let [[min max] (extents key)]
             (if int?
               (u/rand-int-range min max)
               (u/rand-range min max))))]
     {:hidden-sizes [(random-extent :hidden-sizes true)]
      :learning-rate (random-extent :learning-rate false)
      :batch-size (random-extent :batch-size true)
      :epochs (random-extent :epochs true)
      :error-fn error-fn}))
  ([source-params magnitude-range]
   (let [random-extent
         (fn [key source-val int?]
           (let [[min max] (extents key)
                 source (if (vector? source-val) (first source-val) source-val)]
             (if int?
               (u/rand-int-magnitude source magnitude-range min max)
               (u/rand-magnitude source magnitude-range min max))))
         key (rand-nth (keys extents))
         int? (= (type (first (extents key))) Long)
         random-val (random-extent key (source-params key) int?)
         sanitized-val (if (= key :hidden-sizes) [random-val] random-val)]
     (println (str "selected " key))
     (assoc source-params key sanitized-val))))

(defn iterate-hyper
  [data-file test-percentage iterations]
  ; TODO: This is a crude test. Probably just remove it and start over.
  (let [approach (a/approach ag/random-hsl ai/flatten-hsl)
        all-data (cs/read-data approach data-file 1.0)
        training-count (* (- 1.0 test-percentage) (count all-data))
        iteratefn
        (fn []
          (let [shuffled (shuffle all-data)
                [training-data test-data] (split-at training-count shuffled)
                hyper-params (generate-hyper-params)]
            (println hyper-params)
            [hyper-params
             (evaluate-hyper-params training-data test-data hyper-params)]))]
    (repeatedly iterations iteratefn)))

;
; ☞ ☞ ☞    M E T A H E U R I S T I C S
;
; Accepting worse solutions is a fundamental property of metaheuristics
; because it allows for a more extensive search for the optimal solution.
;
; Candidates:
;   - Simulated annealing https://en.wikipedia.org/wiki/Simulated_annealing
;   - Iterated local search https://en.wikipedia.org/wiki/Iterated_local_search
;   - Particle swarm https://en.wikipedia.org/wiki/Particle_swarm_optimization
;
; Also, read the Handbook of metaheuristics, it's in yer Dropbox.

;(iterate-hyper "TR-I3-O1-RAND.clj" 0.2 1)