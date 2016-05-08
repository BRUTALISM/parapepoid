(ns parapepoid.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mo]))

(m/set-current-implementation :vectorz)

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
