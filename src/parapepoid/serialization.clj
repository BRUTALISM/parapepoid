(ns parapepoid.serialization
  (:require [clojure.string :as str]
            [clojure.java.io :as io])
  (:import (java.io PushbackReader FileNotFoundException)))

(def global-prefix "resources")

(defn read-file [filename]
  (try
    (with-open [reader (PushbackReader. (io/reader filename))]
      (binding [*read-eval* true]
        (read reader)))
    (catch FileNotFoundException _
      nil)))

(defn write-file [filename data]
  (with-open [writer (io/writer filename)]
    (binding [*out* writer]
      (pr data))))

(defn- training-path [name]
  (str/join "/" [global-prefix "training" name]))

(defn read-training [name]
  (let [path (training-path name)]
    (read-file path)))

(defn write-training [name data]
  (let [path (training-path name)]
    (write-file path data)))