(ns parapepoid.color.learn
  (:require [parapepoid.util :as u]
            [parapepoid.color.core :as c]
            [parapepoid.nn.core :as n]
            [parapepoid.nn.learn :as l]
            [parapepoid.nn.propagation :as p]
            [parapepoid.serialization :as s]
            [thi.ng.color.core :as tc]))

(def flatten-hsl
  "A transducer for unpacking a sequence of colors into a sequence of floating
  point numbers representing HSL values."
  (let [to-hsl (map tc/as-hsla)
        unpack (mapcat #(u/select-values % [:h :s :l]))]
    (comp to-hsl unpack)))

(defn train-network
  "Reads the [color-palette output] training pairs from the given file and
  trains a neural network using the given params."
  [data-file params]
  (let [{:keys [hidden-sizes learning-rate batch-size epochs]} params
        data (s/read-training data-file)
        prepare (map #(vector (into [] flatten-hsl (first %1))
                              (vector (second %1))))
        prepared-data (into [] prepare data)
        ; TODO: Ne koristi sve podatke za trening, odvoji za validation i test.
        input-count (* 3 (count (first (first data))))
        network (n/network (concat [input-count] hidden-sizes [1]))
        trained (l/sgd network prepared-data batch-size learning-rate epochs)]
    trained))

(defn unscientific-test [params]
  (let [trained (train-network "TR-I3-O1-RAND.clj" params)]
    (doseq [_ (range 20)]
      (prn (p/propagate-forward trained (into [] flatten-hsl
                                              (repeatedly 3 c/random-hsl)))))))

(unscientific-test {:hidden-sizes [4]
                    :learning-rate 0.5
                    :batch-size 10
                    :epochs 30})

; TODO: pronadji najbolje meta-parametre za zadate trening podatke