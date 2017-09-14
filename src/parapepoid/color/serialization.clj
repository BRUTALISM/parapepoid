(ns parapepoid.color.serialization
  (:require [parapepoid.serialization :as s]
            [parapepoid.approach.core :as a]))

(defn training-mapping
  "Returns a transducer which maps a sequence of [color-palette, nn-output]
  pairs into a sequence of [nn-inputs, nn-outputs], using the given individual
  palette transducer xpalette."
  [xpalette]
  (map (fn [[palette output]] (vector (into [] xpalette palette)
                                      (vector output)))))

(defn prepare-data
  "Reads training data from the given file and returns prepared data using
  the given approach."
  [approach filename]
  (let [data (s/read-training filename)]
    (into [] (training-mapping (a/input-mapping approach)) data)))

(defn read-data
  "Using the given approach, reads color palette test data from the file with
  the given filename and returns shuffled data separated into training and test
  data using the given percentage."
  [approach filename test-percentage]
  (let [all-data (prepare-data approach filename)
        training-count (int (* (- 1.0 test-percentage) (count all-data)))
        shuffled (shuffle all-data)]
    (println "Read" (count all-data) "records, using" training-count
             "as training.")
    (split-at training-count shuffled)))